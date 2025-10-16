"""
Solana client abstraction for blockchain operations.
"""

import asyncio
import json
import logging
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
from solders.solders import EncodedConfirmedTransactionWithStatusMeta, Signature
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


class SolanaClient:
    """Abstraction for Solana RPC client operations."""

    def __init__(self, rpc_endpoint: str):
        """Initialize Solana client with RPC endpoint.

        Args:
            rpc_endpoint: URL of the Solana RPC endpoint
        """
        self.rpc_endpoint = rpc_endpoint
        self._client = None
        self._cached_blockhash: Hash | None = None
        self._blockhash_lock = asyncio.Lock()
        self._blockhash_updater_task = asyncio.create_task(
            self.start_blockhash_updater()
        )

    async def start_blockhash_updater(self, interval: float = 5.0):
        """Start background task to update recent blockhash."""
        while True:
            try:
                blockhash = await self.get_latest_blockhash()
                async with self._blockhash_lock:
                    self._cached_blockhash = blockhash
            except Exception as e:
                logger.warning(f"Blockhash fetch failed: {e!s}")
            finally:
                await asyncio.sleep(interval)

    async def get_cached_blockhash(self) -> Hash:
        """Return the most recently cached blockhash."""
        async with self._blockhash_lock:
            if self._cached_blockhash is None:
                raise RuntimeError("No cached blockhash available yet")
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
        """Close the client connection and stop the blockhash updater."""
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

        Returns:
            Transaction signature.
        """
        client = await self.get_client()

        logger.info(
            f"Priority fee in microlamports: {priority_fee if priority_fee else 0}"
        )

        # Add compute budget instructions if applicable
        if priority_fee is not None or compute_unit_limit is not None:
            fee_instructions = []

            # Set compute unit limit (use provided value or default to 85,000)
            cu_limit = compute_unit_limit if compute_unit_limit is not None else 85_000
            fee_instructions.append(set_compute_unit_limit(cu_limit))

            # Set priority fee if provided
            if priority_fee is not None:
                fee_instructions.append(set_compute_unit_price(priority_fee))

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


    @dataclass
    class ConfirmationResult:
        success: bool
        tx:Signature
        error_message: str | None = None
        fees_raw: int | None = None  # Transaction fees in lamports

        def __str__(self) -> str:
            """String representation of confirmation result."""
            result = f"ConfirmationResult(success={self.success}"
            if self.tx:
                result += f", tx='{self.tx}'"
            if self.error_message:
                result += f", error_message='{self.error_message}'"
            if self.fees_raw is not None:
                from core.pubkeys import LAMPORTS_PER_SOL
                fees_sol = self.fees_raw / LAMPORTS_PER_SOL
                result += f", fees_raw={self.fees_raw} lamports ({fees_sol:.6f} SOL)"
            result += ")"
            return result

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

        # Extract fees from the transaction (always try to get them)
        fees = None
        try:
            tx = await client.get_transaction(signature, commitment=commitment)
            if tx and tx.value and tx.value.transaction and tx.value.transaction.meta:
                fees = tx.value.transaction.meta.fee
        except Exception as e:
            logging.info(
                "client.confirm_transaction - failed to extract fees from transaction: %s",
                signature,
            )
            logging.exception(e)

        # Try to extract an error from the confirm response
        if resp.value[0].err:
            # Here I get the transaction anyway so that I can have the logs and extract the error from them
            error_string = str(resp.value[0].err)
            try:
                if tx and tx.value and tx.value.transaction and tx.value.transaction.meta:
                    if tx.value.transaction.meta.log_messages:
                        error_string = str(tx.value.transaction.meta.log_messages)
            except BaseException as e:
                logging.info(
                    "client.confirm_transaction - got exception while getting transaction that failed to get its log messages. Ignoring and will use the following for error extraction: %s",
                    resp,
                )
                logging.exception(e)

            return SolanaClient.ConfirmationResult(
                success=False, tx=signature, error_message=error_string, fees_raw=fees) 

        return SolanaClient.ConfirmationResult(success=True, tx=signature, error_message=None, fees_raw=fees)

    @retry(
        reraise=True,
        wait=wait_fixed(2),
        stop=stop_after_attempt(5),
        retry=retry_if_not_exception_type(UnconfirmedTxError),
        after=after_log(logging.getLogger(), logging.INFO),
    )
    async def get_transaction(self, signature: str) -> EncodedConfirmedTransactionWithStatusMeta:
        """Fetch a transaction by signature.

        Args:
            signature: Transaction signature

        Returns:
            Parsed RPC response dictionary or None on failure
        """
        client = await self.get_client()
        # Use jsonParsed encoding to access meta fields easily
        resp = await client.get_transaction(
            signature,
            encoding="jsonParsed",
            max_supported_transaction_version=0,
            commitment=Confirmed,
        )
        return resp.value


    def compute_token_balance_change(
        self, tx: dict[str, Any], owner: Pubkey, mint: Pubkey
    ) -> int:
        """Compute token balance change for a specific owner and mint.

        Args:
            tx: Transaction data with meta information
            owner: Owner public key
            mint: Token mint address

        Returns:
            Token balance change in raw units
        """
        if not tx or not tx.get("meta"):
            return 0

        meta = tx["meta"]
        pre_balances = meta.get("preTokenBalances", [])
        post_balances = meta.get("postTokenBalances", [])

        if not pre_balances and not post_balances:
            logger.warning(
                f"compute_token_balance_change() - no pre/post token balances for tx={tx.get('transaction', {}).get('signatures', ['unknown'])[0]}"
            )
            return 0

        token_pre_balance = 0
        token_post_balance = 0

        # Find pre-balance
        if pre_balances:
            for balance in pre_balances:
                if balance.get("owner") == str(owner) and balance.get("mint") == str(mint):
                    token_pre_balance = int(balance["uiTokenAmount"]["amount"])
                    break

        # Find post-balance
        if post_balances:
            for balance in post_balances:
                if balance.get("owner") == str(owner) and balance.get("mint") == str(mint):
                    token_post_balance = int(balance["uiTokenAmount"]["amount"])
                    break

        return token_post_balance - token_pre_balance

    def compute_sol_balance_change(self, tx: dict[str, Any], owner: Pubkey) -> int:
        """Compute SOL balance change for a specific owner.

        Args:
            tx: Transaction data with meta information
            owner: Owner public key

        Returns:
            SOL balance change in lamports
        """
        if not tx or not tx.get("meta"):
            return 0

        meta = tx["meta"]
        pre_balances = meta.get("preBalances", [])
        post_balances = meta.get("postBalances", [])
        account_keys = tx.get("transaction", {}).get("message", {}).get("accountKeys", [])

        # Find the account index for the owner
        owner_index = None
        for i, account_key in enumerate(account_keys):
            if account_key == str(owner):
                owner_index = i
                break

        if owner_index is None:
            logger.warning(f"Owner {owner} not found in transaction account keys")
            return 0

        if owner_index >= len(pre_balances) or owner_index >= len(post_balances):
            logger.warning(f"Account index {owner_index} out of range for balance arrays")
            return 0

        pre_balance = int(pre_balances[owner_index])
        post_balance = int(post_balances[owner_index])

        return post_balance - pre_balance



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
