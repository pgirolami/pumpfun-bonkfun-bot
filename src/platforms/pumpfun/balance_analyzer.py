"""
Pump.fun balance analyzer for transaction breakdown.
"""

from solders.pubkey import Pubkey

from core.client import HELIUS_TIP_ACCOUNTS
from core.pubkeys import LAMPORTS_PER_SOL, TOKEN_DECIMALS
from interfaces.core import BalanceAnalyzer, BalanceChangeResult, TokenInfo
from solders.solders import EncodedConfirmedTransactionWithStatusMeta
from utils.logger import get_logger

logger = get_logger(__name__)


class PumpFunBalanceAnalyzer(BalanceAnalyzer):
    """Pump.fun specific balance analyzer."""

    def analyze_balance_changes(
        self, tx: EncodedConfirmedTransactionWithStatusMeta, token_info: TokenInfo, wallet_pubkey: Pubkey, instruction_accounts: dict[str, Pubkey]
    ) -> BalanceChangeResult:
        """Analyze balance changes for pump.fun trading transactions.

        Args:
            tx: Transaction data with meta information
            token_info: Token information including mint, user, creator, etc.
            wallet_pubkey: The wallet executing the transaction (may differ from token_info.user)

        Returns:
            BalanceChangeResult with balance changes and fees in lamports.
            SOL amounts are negative for buys, positive for sells.
            Token amounts are positive for buys, negative for sells.
        """

        #What happens on a buy, see https://solscan.io/tx/kpFzKZYbDiJ4wxTA7q5sskKveDXSMgerFbg9amEWYBddY5ZEFrtk3jgtV9XDvboxjdYNEyrqSNWE2B4acFhTk2o#solBalanceChange
        #- An associated token account is created with rent exemption so it transfers the minimum amount of SOL in that account (On 2025-10-25, this was 0.00203928 SOL)
        #- The "PumpFun fee account" receives a few hundred lamports
        #- The creator vault receives a few hundred lamports
        #- There is a transaction fee of 0.000025 SOL
        #-> if you sum all these numbers, you get to total amount of SOL transfered out of the wallet's account
        #
        # For our purposes, we don't get about the rent amount since we will get it back so we should ignore it in our computation which means
        # we have to remove it from the total.
        #
        #To get the token balances, you have to rely on token_pre_balances and token_post_balances
        
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
            raise RuntimeError(f"Wallet {wallet_pubkey} not found in transaction account keys")

        sol_amount_raw = int(post_balances[wallet_index]) - int(pre_balances[wallet_index])

        # Calculate rent exemption amount for user's token account
        user_token_account = instruction_accounts["user_token_account"]
        rent_exemption_amount_raw = 0
        
        # Find the user token account in the transaction
        user_token_account_index = None
        for i, account_key in enumerate(tx.transaction.transaction.message.account_keys):
            if account_key.pubkey == user_token_account:
                user_token_account_index = i
                break
        
        # Calculate the actual SOL balance change for the token account
        if user_token_account_index is not None:
            #Purposely reversed to get the amount of SOL that was transferred out of the wallet's account: negative on buys
            rent_exemption_amount_raw = int(pre_balances[user_token_account_index]) - int(post_balances[user_token_account_index])

        # Extract real fees from transaction balance changes
        # Use addresses from instruction_accounts
        creator_vault = instruction_accounts["creator_vault"]
        fee_account = instruction_accounts["fee"]
        
        # Find the account indices
        fee_index = None
        creator_vault_index = None
        bonding_curve_index = None
        
        # Get bonding curve from instruction accounts
        bonding_curve = instruction_accounts.get("bonding_curve")
        
        for i, account_key in enumerate(tx.transaction.transaction.message.account_keys):
            if account_key.pubkey == fee_account:
                fee_index = i
            elif account_key.pubkey == creator_vault:
                creator_vault_index = i
            elif bonding_curve and account_key.pubkey == bonding_curve:
                bonding_curve_index = i
        
        # Calculate net_sol_swap_amount to bonding curve balance change
        # On buy: bonding curve receives SOL => positive change
        bonding_curve_swap_amount_raw = int(post_balances[bonding_curve_index]) - int(pre_balances[bonding_curve_index])        

        # Calculate actual fees from balance changes
        protocol_fee_raw = 0
        creator_fee_raw = 0
        token_swap_amount_raw = 0
        
        protocol_fee_raw = int(post_balances[fee_index]) - int(pre_balances[fee_index])
        
        creator_fee_raw = int(post_balances[creator_vault_index]) - int(pre_balances[creator_vault_index])
        
        # Calculate token amount from user's token account balance changes
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
        
        total_platform_fee_raw = protocol_fee_raw + creator_fee_raw
        
        # Detect tip transfers to Helius tip accounts
        tip_fee_raw = 0
        tip_accounts_set = set(HELIUS_TIP_ACCOUNTS)
        for i, account_key in enumerate(tx.transaction.transaction.message.account_keys):
            if account_key.pubkey in tip_accounts_set:
                if i < len(pre_balances) and i < len(post_balances):
                    tip_balance_change = int(post_balances[i]) - int(pre_balances[i])
                    if tip_balance_change > 0:
                        tip_fee_raw += tip_balance_change
        

        net_sol_swap_amount_raw = -bonding_curve_swap_amount_raw #- total_platform_fee_raw

        unattributed_sol_amount_raw = sol_amount_raw - (net_sol_swap_amount_raw - transaction_fee - tip_fee_raw + rent_exemption_amount_raw)

        if unattributed_sol_amount_raw != 0:
            logger.warning(f"[{str(token_info.mint)[:8]}] Unattributed SOL amount in balance check : {unattributed_sol_amount_raw} lamports in transaction {str(tx.transaction.transaction.signatures[0])}")

        # All this has been checked, rechecked and checked again
        # See https://docs.google.com/spreadsheets/d/1UN6nxlqMq0SU2wCmwCoprO5WGA3pacxuwBaIl9LmnpQ/edit?gid=219123703#gid=219123703

        return BalanceChangeResult(
            token_swap_amount_raw=token_swap_amount_raw,  # Positive for buys, negative for sells
            net_sol_swap_amount_raw=net_sol_swap_amount_raw,  # Negative for buys, positive for sells
            rent_exemption_amount_raw=rent_exemption_amount_raw,
            unattributed_sol_amount_raw=unattributed_sol_amount_raw,
            protocol_fee_raw=protocol_fee_raw,
            creator_fee_raw=creator_fee_raw,
            transaction_fee_raw=transaction_fee,
            tip_fee_raw=tip_fee_raw,
            sol_amount_raw=sol_amount_raw,
        )