"""
LetsBonk balance analyzer for transaction breakdown.
"""

from typing import Any

from solders.pubkey import Pubkey
from solders.solders import EncodedConfirmedTransactionWithStatusMeta

from core.pubkeys import LAMPORTS_PER_SOL, TOKEN_DECIMALS
from interfaces.core import BalanceAnalyzer, BalanceChangeResult, TokenInfo
from platforms.letsbonk.address_provider import LetsBonkAddressProvider
from utils.logger import get_logger

logger = get_logger(__name__)


class LetsBonkBalanceAnalyzer(BalanceAnalyzer):
    """LetsBonk specific balance analyzer."""

    def analyze_balance_changes(
        self, tx: EncodedConfirmedTransactionWithStatusMeta, token_info: TokenInfo, wallet_pubkey: Pubkey, instruction_accounts: dict[str, Pubkey]
    ) -> BalanceChangeResult:
        """Analyze balance changes for LetsBonk trading transactions.

        Args:
            tx: Transaction data with meta information
            token_info: Token information including mint, user, creator, etc.
            wallet_pubkey: The wallet executing the transaction (may differ from token_info.user)

        Returns:
            BalanceChangeResult with balance changes and fees in lamports.
            SOL amounts are negative for buys, positive for sells.
            Token amounts are positive for buys, negative for sells.
        """
        if not tx or not tx.transaction or not tx.transaction.meta:
            logger.warning("No transaction or meta found")
            return BalanceChangeResult()

        meta = tx.transaction.meta
        pre_balances = meta.pre_balances
        post_balances = meta.post_balances

        # Get transaction fee from meta
        transaction_fee = meta.fee or 0

        # Calculate SOL balance change for wallet
        wallet_index = None
        for i, account_key in enumerate(tx.transaction.transaction.message.account_keys):
            if account_key.pubkey == wallet_pubkey:
                wallet_index = i
                break

        if wallet_index is None:
            logger.warning(f"Wallet {wallet_pubkey} not found in transaction account keys")
            return BalanceChangeResult()

        sol_amount_raw = int(post_balances[wallet_index]) - int(pre_balances[wallet_index])

        # Calculate rent exemption amount for user's token account
        user_token_account = instruction_accounts["user_base_token"]
        rent_exemption_amount_raw = 0
        
        # Find the user token account in the transaction
        user_token_account_index = None
        for i, account_key in enumerate(tx.transaction.transaction.message.account_keys):
            if account_key.pubkey == user_token_account:
                user_token_account_index = i
                break
        
        # Calculate the actual SOL balance change for the token account
        if user_token_account_index is not None:
            rent_exemption_amount_raw = int(post_balances[user_token_account_index]) - int(pre_balances[user_token_account_index])
            logger.info(f"User token account SOL balance change: {rent_exemption_amount_raw} lamports")

        # For LetsBonk (DEX), extract real fees from vault balance changes
        # Use addresses from instruction_accounts
        pool_state = instruction_accounts["pool_state"]
        base_vault = instruction_accounts["base_vault"]
        quote_vault = instruction_accounts["quote_vault"]
        
        # Find the account indices for vaults
        base_vault_index = None
        quote_vault_index = None
        
        for i, account_key in enumerate(tx.transaction.transaction.message.account_keys):
            if account_key.pubkey == base_vault:
                base_vault_index = i
            elif account_key.pubkey == quote_vault:
                quote_vault_index = i
        
        if base_vault_index is None or quote_vault_index is None:
            raise RuntimeError(f"Could not find vault account indices for base_vault={base_vault}, quote_vault={quote_vault}")
        
        # Calculate vault balance changes
        base_vault_change = int(post_balances[base_vault_index]) - int(pre_balances[base_vault_index])
        quote_vault_change = int(post_balances[quote_vault_index]) - int(pre_balances[quote_vault_index])
        
        # Calculate platform fee from vault changes
        # For DEX, the fee is the difference between what user spent/received and what vaults received/sent, minus transaction fee
        # BROKEN BECAUSE OF RENT EXEMPTION, didn't spend time fixing it 
        if sol_amount_raw < 0:  # Buy transaction
            # User spent SOL, vault received less due to fees
            vault_sol_received = abs(quote_vault_change)  # Quote vault receives SOL
            
            platform_fee_raw = abs(sol_amount_raw) - vault_sol_received - transaction_fee
        else:  # Sell transaction
            # User received SOL, vault sent more due to fees
            vault_sol_sent = abs(quote_vault_change)  # Quote vault sends SOL
            platform_fee_raw = vault_sol_sent - sol_amount_raw - transaction_fee
                
        # Calculate token amount from user's token account balance changes
        token_swap_amount_raw = 0
        if user_token_account_index is not None:
            # Use token balances instead of SOL balances for token amounts
            token_pre_balances = meta.pre_token_balances or []
            token_post_balances = meta.post_token_balances or []
            
            # Find the user's token account in token balances
            user_token_pre_balance = 0
            user_token_post_balance = 0
            
            for token_balance in token_pre_balances:
                if (hasattr(token_balance, 'account_index') and 
                    token_balance.account_index == user_token_account_index):
                    user_token_pre_balance = int(token_balance.ui_token_amount.amount)
                    break
            
            for token_balance in token_post_balances:
                if (hasattr(token_balance, 'account_index') and 
                    token_balance.account_index == user_token_account_index):
                    user_token_post_balance = int(token_balance.ui_token_amount.amount)
                    break
            
            token_swap_amount_raw = user_token_post_balance - user_token_pre_balance
            logger.info(f"User token account balance change: {user_token_pre_balance} -> {user_token_post_balance} = {token_swap_amount_raw} raw units")
                    
        # Determine transaction type for logging
        tx_type = "buy" if sol_amount_raw < 0 else "sell"
        
        logger.info(
            f"LetsBonk {tx_type} analysis: "
            f"user_sol_change={sol_amount_raw/LAMPORTS_PER_SOL:.6f} SOL, "
            f"base_vault_change={base_vault_change/LAMPORTS_PER_SOL:.6f} SOL, "
            f"quote_vault_change={quote_vault_change/LAMPORTS_PER_SOL:.6f} SOL, "
            f"platform_fee={platform_fee_raw/LAMPORTS_PER_SOL:.6f} SOL, "
            f"rent_exemption={rent_exemption_amount_raw/LAMPORTS_PER_SOL:.6f} SOL, "
            f"token_swap_amount={token_swap_amount_raw/TOKEN_DECIMALS:.6f} tokens"
        )
        
        return BalanceChangeResult(
            sol_amount_raw=sol_amount_raw,  # Negative for buys, positive for sells
            rent_exemption_amount_raw=rent_exemption_amount_raw, #positive for buys
            sol_swap_amount_raw=sol_swap_amount_raw,
            platform_fee_raw=platform_fee_raw,
            transaction_fee_raw=transaction_fee,
            token_swap_amount_raw=token_swap_amount_raw,  # Positive for buys, negative for sells
        )
