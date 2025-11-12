"""
Solana client abstraction for blockchain operations.
"""

import asyncio
import base64
import json
import logging
import random
import struct
from typing import Any

import aiohttp
from dataclasses import dataclass
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Commitment, Confirmed, Processed
from solana.rpc.core import UnconfirmedTxError
from solana.rpc.types import TxOpts
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.hash import Hash
from solders.instruction import Instruction
from solders.keypair import Keypair
from solders.message import Message
from solders.pubkey import Pubkey
from solders.solders import EncodedConfirmedTransactionWithStatusMeta
from solders.signature import Signature
from solders.system_program import TransferParams, transfer
from solders.transaction import Transaction

from utils.logger import get_logger
from tenacity import (
    after_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_fixed,
)

logger = get_logger(__name__)

# Helius tip account addresses
HELIUS_TIP_ACCOUNTS = [
    Pubkey.from_string("4ACfpUFoaSD9bfPdeu6DBt89gB6ENTeHBXCAi87NhDEE"),
    Pubkey.from_string("D2L6yPZ2FmmmTKPgzaMKdhu6EWZcTpLy1Vhx8uvZe7NZ"),
    Pubkey.from_string("9bnz4RShgq1hAnLnZbP8kbgBg1kEmcJBYQq3gQbmnSta"),
    Pubkey.from_string("5VY91ws6B2hMmBFRsXkoAAdsPHBJwRfBht4DXox3xkwn"),
    Pubkey.from_string("2nyhqdwKcJZR2vcqCyrYsaPVdAnFoJjiksCXJ7hfEYgD"),
    Pubkey.from_string("2q5pghRs6arqVjRvT5gfgWfWcHWmw1ZuCzphgd5KfWGJ"),
    Pubkey.from_string("wyvPkWjVZz1M8fHQnMMCDTQDbkManefNNhweYk5WkcF"),
    Pubkey.from_string("3KCKozbAaF75qEU33jtzozcJ29yJuaLJTy2jFdzUY8bT"),
    Pubkey.from_string("4vieeGHPYPG2MmyPRcYjdiDmmhN3ww7hsFNap8pVN3Ey"),
    Pubkey.from_string("4TQLFNWK8AovT1gFvda5jfw2oJeRMKEmw7aH6MGBJ3or"),
]


def set_loaded_accounts_data_size_limit(bytes_limit: int) -> Instruction:
    """
    Create SetLoadedAccountsDataSizeLimit instruction to reduce CU consumption.

    By default, Solana transactions can load up to 64MB of account data,
    costing 16k CU (8 CU per 32KB). Setting a lower limit reduces CU
    consumption and improves transaction priority.

    NOTE: CU savings are NOT visible in "consumed CU" metrics, which only
    show execution CU. The 16k CU loaded accounts overhead is counted
    separately for transaction priority/cost calculation.

    Args:
        bytes_limit: Max account data size in bytes (e.g., 512_000 = 512KB)

    Returns:
        Compute Budget instruction with discriminator 4

    Reference:
        https://www.anza.xyz/blog/cu-optimization-with-setloadedaccountsdatasizelimit
    """
    COMPUTE_BUDGET_PROGRAM = Pubkey.from_string(
        "ComputeBudget111111111111111111111111111111"
    )

    data = struct.pack("<BI", 4, bytes_limit)
    return Instruction(COMPUTE_BUDGET_PROGRAM, data, [])


class SolanaClient:
    """Abstraction for Solana RPC client operations."""

    tip_amount_lamports = int(0.000005*1_000_000_000)

    def __init__(self, rpc_config: dict, blockhash_update_interval: float = 10.0):
        """Initialize Solana client with RPC configuration.

        Args:
            rpc_config: Dictionary containing:
                - rpc_endpoint: URL of the Solana RPC endpoint
                - wss_endpoint: URL of the WebSocket endpoint
                - send_method: Either "solana" (standard RPC) or "helius_sender"
                - helius_sender (optional): Configuration dict with:
                    - routing: "dual" or "swqos_only"
                    - tip_amount_sol: Tip amount in SOL (defaults based on routing)
                    - endpoint (optional): Override endpoint (defaults to London)
            blockhash_update_interval: Interval in seconds to update cached blockhash
        """
        self.rpc_endpoint = rpc_config["rpc_endpoint"]
        self.wss_endpoint = rpc_config["wss_endpoint"]
        self.send_method = rpc_config.get("send_method", "solana")
        self._helius_sender_config = rpc_config.get("helius_sender")
        
        # Initialize Helius Sender attributes (always initialized, even if not used)
        self._helius_session: aiohttp.ClientSession | None = None
        self._helius_ping_task: asyncio.Task | None = None
        
        # Setup Helius Sender if enabled
        if self.send_method == "helius_sender":
            if not self._helius_sender_config:
                self._helius_sender_config = {}
            routing = self._helius_sender_config.get("routing", "swqos_only")
            # Use HTTPS endpoint as per Helius documentation
            base_endpoint = self._helius_sender_config.get(
                "endpoint", "http://lon-sender.helius-rpc.com/fast"
            )
            # Append routing query parameter if swqos_only
            if routing == "swqos_only":
                # Add swqos_only parameter if not already present
                if "swqos_only" not in base_endpoint:
                    separator = "&" if "?" in base_endpoint else "?"
                    self._helius_endpoint = f"{base_endpoint}{separator}swqos_only=true"
                else:
                    self._helius_endpoint = base_endpoint
            else:
                self._helius_endpoint = base_endpoint
            # Construct ping endpoint URL from main endpoint (remove query params for ping)
            ping_base = self._helius_endpoint.split("?")[0]
            if "/fast" in ping_base:
                self._helius_ping_endpoint = ping_base.replace("/fast", "/ping")
            else:
                # Fallback: append /ping if /fast not found
                ping_base = ping_base.rstrip("/")
                self._helius_ping_endpoint = f"{ping_base}/ping"
            # Default tip amounts based on routing
            if routing == "dual":
                self.tip_amount_lamports = int(
                    self._helius_sender_config.get("tip_amount_sol", 0.001)
                    * 1_000_000_000
                )
            else:  # swqos_only (default)
                self.tip_amount_lamports = int(
                    self._helius_sender_config.get("tip_amount_sol", 0.000005)
                    * 1_000_000_000
                )
            logger.info(
                f"Helius Sender enabled: endpoint={self._helius_endpoint}, "
                f"routing={routing}, tip={self.tip_amount_lamports / 1_000_000_000} SOL"
            )
        
        self._client = None
        self._cached_blockhash: Hash | None = None
        self._blockhash_lock = asyncio.Lock()
        self._blockhash_update_interval = blockhash_update_interval
        self._blockhash_updater_task = asyncio.create_task(
            self.start_blockhash_updater()
        )
        
        # Start Helius Sender connection warming if enabled
        if self.send_method == "helius_sender":
            self._helius_session = aiohttp.ClientSession()
            self._helius_ping_task = asyncio.create_task(
                self._start_helius_ping_loop()
            )
            logger.info("Helius Sender connection warming started")

    async def start_blockhash_updater(self, interval: float | None = None):
        """Start background task to update recent blockhash."""
        if interval is None:
            interval = self._blockhash_update_interval
            
        logger.info(f"Starting blockhash updater with {interval}s interval")
        while True:
            try:
                blockhash = await self.get_latest_blockhash()
                async with self._blockhash_lock:
                    self._cached_blockhash = blockhash
                logger.debug(f"Updated cached blockhash: {blockhash}")
                await asyncio.sleep(interval)
            except Exception as e:
                logger.warning(f"Blockhash fetch failed: {e!s}")
                await asyncio.sleep(2.0)

    async def get_cached_blockhash(self) -> Hash:
        """Return the most recently cached blockhash."""
        async with self._blockhash_lock:
            if self._cached_blockhash is None:
                logger.warning("No cached blockhash available, fetching fresh one...")
                # Fallback to fresh fetch if cache is empty
                blockhash = await self.get_latest_blockhash()
                self._cached_blockhash = blockhash
                logger.info(f"Fetched fresh blockhash: {blockhash}")
            else:
                logger.debug(f"Using cached blockhash: {self._cached_blockhash}")
            return self._cached_blockhash

    async def get_client(self) -> AsyncClient:
        """Get or create the AsyncClient instance.

        Returns:
            AsyncClient instance
        """
        if self._client is None:
            self._client = AsyncClient(self.rpc_endpoint)
        return self._client

    async def close(self):
        """Close the client connection and stop background tasks."""
        # Stop Helius Sender ping task
        if self._helius_ping_task:
            self._helius_ping_task.cancel()
            try:
                await self._helius_ping_task
            except asyncio.CancelledError:
                pass
            self._helius_ping_task = None
        
        # Close Helius Sender HTTP session
        if self._helius_session:
            await self._helius_session.close()
            self._helius_session = None
        
        # Stop blockhash updater
        if self._blockhash_updater_task:
            self._blockhash_updater_task.cancel()
            try:
                await self._blockhash_updater_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.close()
            self._client = None

    async def get_health(self) -> str | None:
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getHealth",
        }
        result = await self.post_rpc(body)
        if result and "result" in result:
            return result["result"]
        return None

    async def get_account_info(self, pubkey: Pubkey) -> dict[str, Any]:
        """Get account info from the blockchain.

        Args:
            pubkey: Public key of the account

        Returns:
            Account info response

        Raises:
            ValueError: If account doesn't exist or has no data
        """
        client = await self.get_client()
        response = await client.get_account_info(
            pubkey, encoding="base64"
        )  # base64 encoding for account data by default
        if not response.value:
            raise ValueError(f"Account {pubkey} not found")
        return response.value

    async def get_token_account_balance(self, token_account: Pubkey) -> int:
        """Get token balance for an account.

        Args:
            token_account: Token account address

        Returns:
            Token balance as integer
        """
        client = await self.get_client()
        response = await client.get_token_account_balance(token_account)
        if response.value:
            return int(response.value.amount)
        return 0

    async def get_sol_balance(self, pubkey: Pubkey) -> int:
        """Get SOL balance for a wallet account.

        Args:
            pubkey: Public key of the wallet

        Returns:
            SOL balance in lamports
        """
        client = await self.get_client()
        response = await client.get_balance(pubkey)
        return response.value

    async def get_latest_blockhash(self) -> Hash:
        """Get the latest blockhash.

        Returns:
            Recent blockhash as string
        """
        client = await self.get_client()
        response = await client.get_latest_blockhash(commitment=Processed)
        return response.value.blockhash

    async def build_and_send_transaction(
        self,
        instructions: list[Instruction],
        signer_keypair: Keypair,
        skip_preflight: bool = True,
        max_retries: int = 3,
        priority_fee: int | None = None,
        compute_unit_limit: int | None = None,
        account_data_size_limit: int | None = None,
    ) -> Signature:
        """
        Send a transaction with optional priority fee and compute unit limit.

        Args:
            instructions: List of instructions to include in the transaction.
            signer_keypair: Keypair to sign the transaction.
            skip_preflight: Whether to skip preflight checks.
            max_retries: Maximum number of retry attempts.
            priority_fee: Optional priority fee in microlamports.
            compute_unit_limit: Optional compute unit limit. Defaults to 85,000 if not provided.
            account_data_size_limit: Optional account data size limit in bytes (e.g., 512_000).
                                    Reduces CU cost from 16k to ~128 CU. Must be first instruction.

        Returns:
            Transaction signature.
        """
        client = await self.get_client()

        logger.info(
            f"Priority fee in microlamports: {priority_fee if priority_fee else 0}"
        )

        # Add compute budget instructions if applicable
        if (
            priority_fee is not None
            or compute_unit_limit is not None
            or account_data_size_limit is not None
        ):
            fee_instructions = []

            if account_data_size_limit is not None:
                fee_instructions.append(
                    set_loaded_accounts_data_size_limit(account_data_size_limit)
                )
                logger.info(f"Account data size limit: {account_data_size_limit} bytes")

            # Set compute unit limit (use provided value or default to 85,000)
            cu_limit = compute_unit_limit if compute_unit_limit is not None else 85_000
            fee_instructions.append(set_compute_unit_limit(cu_limit))

            # Set priority fee if provided
            if priority_fee is not None:
                fee_instructions.append(set_compute_unit_price(priority_fee))

        # If Helius Sender is enabled, add tip instruction and send via Sender
        if self.send_method == "helius_sender":
            tip_instruction = self._add_tip_instruction(
                signer_keypair.pubkey(), self.tip_amount_lamports
            )
            # CRITICAL: Order is compute budget → tip → user instructions
            instructions = fee_instructions + [tip_instruction] + instructions
            recent_blockhash = await self.get_cached_blockhash()
            message = Message(instructions, signer_keypair.pubkey())
            transaction = Transaction([signer_keypair], message, recent_blockhash)
            
            for attempt in range(max_retries):
                try:
                    return await self._send_via_helius_sender(transaction)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.exception(
                            f"Failed to send transaction via Helius Sender after {max_retries} attempts"
                        )
                        raise
                    wait_time = 2**attempt
                    logger.warning(
                        f"Transaction attempt {attempt + 1} failed: {e!s}, retrying in {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)
        else:
            # Standard flow: just prepend fee instructions
            instructions = fee_instructions + instructions
            recent_blockhash = await self.get_cached_blockhash()
            message = Message(instructions, signer_keypair.pubkey())
            transaction = Transaction([signer_keypair], message, recent_blockhash)

            for attempt in range(max_retries):
                try:
                    tx_opts = TxOpts(
                        skip_preflight=skip_preflight, preflight_commitment=Processed
                    )
                    response = await client.send_transaction(transaction, tx_opts)
                    return response.value

                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.exception(
                            f"Failed to send transaction after {max_retries} attempts"
                        )
                        raise

                    wait_time = 2**attempt
                    logger.warning(
                        f"Transaction attempt {attempt + 1} failed: {e!s}, retrying in {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)

    def _add_tip_instruction(
        self, from_pubkey: Pubkey, tip_amount_lamports: int
    ) -> Instruction:
        """Create a tip transfer instruction to a random Helius tip account.

        Args:
            from_pubkey: Public key of the wallet sending the tip
            tip_amount_lamports: Tip amount in lamports

        Returns:
            Instruction for SOL transfer to tip account
        """
        tip_account = random.choice(HELIUS_TIP_ACCOUNTS)
        return transfer(
            TransferParams(
                from_pubkey=from_pubkey, to_pubkey=tip_account, lamports=tip_amount_lamports
            )
        )

    async def _send_via_helius_sender(self, transaction: Transaction) -> Signature:
        """Send transaction via Helius Sender endpoint.

        Args:
            transaction: Signed transaction to send

        Returns:
            Transaction signature

        Raises:
            Exception: If the transaction fails to send
        """
        # Serialize transaction to base64
        serialized = bytes(transaction)
        base64_tx = base64.b64encode(serialized).decode("utf-8")

        # Prepare JSON-RPC 2.0 request
        request_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendTransaction",
            "params": [
                base64_tx,
                {
                    "encoding": "base64",
                    "skipPreflight": True,
                    "maxRetries": 0,
                },
            ],
        }

        # Use persistent session for connection warming
        if not self._helius_session:
            self._helius_session = aiohttp.ClientSession()
        
        try:
            async with self._helius_session.post(
                self._helius_endpoint,
                json=request_body,
                timeout=aiohttp.ClientTimeout(10),
            ) as response:
                # Try to get response body even on error for better debugging
                try:
                    result = await response.json()
                except Exception:
                    # If JSON parsing fails, get text response
                    text_result = await response.text()
                    logger.error(
                        f"Helius Sender HTTP {response.status} error. Response: {text_result[:500]}"
                    )
                    response.raise_for_status()  # Will raise HTTPException
                
                # Check for HTTP errors
                if response.status != 200:
                    error_detail = result.get("error", {}) if isinstance(result, dict) else str(result)
                    logger.error(
                        f"Helius Sender HTTP {response.status} error. Response: {error_detail}"
                    )
                    response.raise_for_status()
                
                # Check for JSON-RPC errors
                if "error" in result:
                    error_msg = result["error"].get("message", "Unknown error")
                    error_code = result["error"].get("code", "N/A")
                    raise Exception(
                        f"Helius Sender JSON-RPC error (code {error_code}): {error_msg}"
                    )

                if "result" not in result:
                    raise Exception(f"Helius Sender response missing result: {result}")

                signature_str = result["result"]
                logger.info(f"Transaction sent via Helius Sender: {signature_str}")
                return Signature.from_string(signature_str)
        except aiohttp.ClientResponseError as e:
            # Include more context about the HTTP error
            logger.error(
                f"Helius Sender HTTP error {e.status}: {e.message}. "
                f"URL: {self._helius_endpoint}"
            )
            raise Exception(f"Helius Sender HTTP {e.status}: {e.message}")
        except aiohttp.ClientError as e:
            logger.error(f"Helius Sender client error: {e!s}")
            raise Exception(f"Helius Sender connection error: {e!s}")

    async def _ping_helius_sender(self) -> bool:
        """Send a ping to the Helius Sender endpoint to keep connection warm.

        Returns:
            True if ping succeeded, False otherwise
        """
        if not self._helius_session:
            return False
        
        try:
            async with self._helius_session.get(
                self._helius_ping_endpoint,
                timeout=aiohttp.ClientTimeout(5),
            ) as response:
                response.raise_for_status()
                logger.debug(f"Helius Sender ping successful: {response.status}")
                return True
        except Exception as e:
            logger.debug(f"Helius Sender ping failed: {e!s}")
            return False

    async def _start_helius_ping_loop(self, interval: float = 45.0) -> None:
        """Start background ping loop to maintain warm connections.

        Args:
            interval: Seconds between pings (default 45s, within 30-60s range)
        """
        logger.info(f"Starting Helius Sender ping loop with {interval}s interval")
        
        # Initial ping to warm connection immediately
        await self._ping_helius_sender()
        
        while True:
            try:
                await asyncio.sleep(interval)
                await self._ping_helius_sender()
            except asyncio.CancelledError:
                logger.debug("Helius Sender ping loop cancelled")
                break
            except Exception as e:
                logger.warning(f"Error in Helius Sender ping loop: {e!s}")
                # Continue pinging even on error
                await asyncio.sleep(5)


    @dataclass
    class TransactionMetaInfo:
        """Transaction metadata information extracted from transaction object."""
        block_time: int | None = None  # Block time in Unix timestamp
        block_slot: int | None = None  # Block slot number
        success: bool = True  # Whether transaction succeeded
        error_message: str | None = None  # Error message if failed
        log_messages: list[str] | None = None  # Transaction log messages

    @dataclass
    class ConfirmationResult:
        success: bool
        tx:Signature
        error_message: str | None = None
        block_ts: int | None = None  # Unix epoch milliseconds

        def __str__(self) -> str:
            """String representation of confirmation result."""
            result = f"ConfirmationResult(success={self.success}"
            if self.tx:
                result += f", tx='{self.tx}'"
            if self.error_message:
                result += f", error_message='{self.error_message}'"
            result += ")"
            return result

    def analyze_transaction_meta(
        self, tx: EncodedConfirmedTransactionWithStatusMeta
    ) -> "SolanaClient.TransactionMetaInfo":
        """Analyze transaction metadata and extract information.

        Args:
            tx: Transaction object with meta information

        Returns:
            TransactionMetaInfo: Dataclass with extracted transaction metadata
        """
        block_time = None
        block_slot = None
        success = True
        error_message = None
        log_messages = None

        if tx:
            # Extract block time
            if hasattr(tx, 'block_time') and tx.block_time:
                block_time = int(tx.block_time)
            
            # Extract block slot
            if hasattr(tx, 'slot'):
                block_slot = int(tx.slot)

            # Extract error and log messages from transaction meta
            if tx.transaction and tx.transaction.meta:
                meta = tx.transaction.meta
                
                # Check if transaction succeeded
                if meta.err:
                    success = False
                    error_message = str(meta.err)
                
                # Extract log messages
                if meta.log_messages:
                    log_messages = list(meta.log_messages)

        return SolanaClient.TransactionMetaInfo(
            block_time=block_time,
            block_slot=block_slot,
            success=success,
            error_message=error_message,
            log_messages=log_messages,
        )

    async def confirm_transaction(
        self, signature: Signature, commitment: Commitment = Confirmed
    ) -> "SolanaClient.ConfirmationResult":
        """Wait for transaction confirmation and extract error details if any.

        Args:
            signature: Transaction signature to confirm.
            commitment: Confirmation commitment level (e.g., "processed", "confirmed").

        Returns:
            ConfirmationResult: Dataclass containing:
                - success: True if confirmed without RPC error in the response.
                - error_message: Enriched error details if an error was present;
                  includes transaction logMessages when available.
        """
        client = await self.get_client()

        # Wait for confirmation and inspect response for errors
        resp = await client.confirm_transaction(
            signature, commitment=commitment, sleep_seconds=1
        )

        # Extract error from confirmation response
        error_string = None
        if resp.value[0].err:
            error_string = str(resp.value[0].err)
        
        # Fetch transaction for additional details (error logs and block time)
        block_time = None
        try:
            tx = await client.get_transaction(signature, commitment=commitment)
            if tx and tx.value:
                # Use analyze_transaction_meta to extract information
                meta_info = self.analyze_transaction_meta(tx.value)
                block_time = meta_info.block_time
                
                # Use detailed error messages from transaction if available
                if meta_info.error_message:
                    error_string = meta_info.error_message
                elif meta_info.log_messages and error_string:
                    # Prefer log messages if available
                    error_string = "\n".join(meta_info.log_messages)
        except BaseException as e:
            logging.info(
                "client.confirm_transaction - got exception while getting transaction details: %s",
                e
            )

        return SolanaClient.ConfirmationResult(
            success=not resp.value[0].err, 
            tx=signature, 
            error_message=error_string,
            block_ts=block_time
        )

    @retry(
        reraise=True,
        wait=wait_fixed(2),
        stop=stop_after_attempt(5),
        retry=retry_if_not_exception_type(UnconfirmedTxError),
        after=after_log(logging.getLogger(), logging.INFO),
    )
    async def get_transaction(self, signature: str | Signature) -> EncodedConfirmedTransactionWithStatusMeta:
        """Fetch a transaction by signature.

        Args:
            signature: Transaction signature (string or Signature object)

        Returns:
            Parsed RPC response dictionary or None on failure
        """
        client = await self.get_client()
        # Convert string to Signature object if needed
        if isinstance(signature, str):
            signature = Signature.from_string(signature)
        # Use jsonParsed encoding to access meta fields easily
        resp = await client.get_transaction(
            signature,
            encoding="jsonParsed",
            max_supported_transaction_version=0,
            commitment=Confirmed,
        )
        return resp.value

    async def post_rpc(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """
        Send a raw RPC request to the Solana node.

        Args:
            body: JSON-RPC request body.

        Returns:
            Optional[Dict[str, Any]]: Parsed JSON response, or None if the request fails.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.rpc_endpoint,
                    json=body,
                    timeout=aiohttp.ClientTimeout(10),  # 10-second timeout
                ) as response:
                    response.raise_for_status()
                    return await response.json()
        except aiohttp.ClientError:
            logger.exception("RPC request failed")
            return None
        except json.JSONDecodeError:
            logger.exception("Failed to decode RPC response")
            return None
