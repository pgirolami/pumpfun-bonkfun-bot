"""
Pump.Fun implementation of AddressProvider interface.

This module provides all pump.fun-specific addresses and PDA derivations
by implementing the AddressProvider interface.
"""

from dataclasses import dataclass
from typing import Final

from solders.pubkey import Pubkey
from spl.token.instructions import get_associated_token_address

from core.pubkeys import SystemAddresses
from interfaces.core import AddressProvider, Platform, TokenInfo


@dataclass
class PumpFunAddresses:
    """Pump.fun program addresses."""

    PROGRAM: Final[Pubkey] = Pubkey.from_string(
        "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
    )
    GLOBAL: Final[Pubkey] = Pubkey.from_string(
        "4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf"
    )
    EVENT_AUTHORITY: Final[Pubkey] = Pubkey.from_string(
        "Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1"
    )
    FEE: Final[Pubkey] = Pubkey.from_string(
        "CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM"
    )
    # Mayhem mode fee recipient (hardcoded to avoid RPC calls)
    # To check if this address is up-to-date, fetch Global account data at offset 483
    # from the pump.fun Global account: 4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf
    MAYHEM_FEE: Final[Pubkey] = Pubkey.from_string(
        "GesfTA3X2arioaHp8bbKdjG9vJtskViWACZoYvxp4twS"
    )
    LIQUIDITY_MIGRATOR: Final[Pubkey] = Pubkey.from_string(
        "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"
    )
    FEE_PROGRAM: Final[Pubkey] = Pubkey.from_string(
        "pfeeUxB6jkeY1Hxd7CsFCAjcbHA9rWtchMGdZ6VojVZ"
    )
    
    # Fee recipients from global config (main fee_recipient + 7 fee_recipients, 8 total)
    # Set at bot startup via set_fee_recipients()
    _fee_recipients: list[Pubkey] | None = None
    
    @classmethod
    def set_fee_recipients(cls, fee_recipients: list[Pubkey]) -> None:
        """Set the fee recipients from global config.
        
        Args:
            fee_recipients: List of 8 fee recipient pubkeys (main fee_recipient + 7 fee_recipients)
        """
        cls._fee_recipients = fee_recipients
    
    @classmethod
    def get_fee_recipients(cls) -> list[Pubkey]:
        """Get all fee recipients (main fee_recipient + 7 fee_recipients).
        
        Returns:
            List of 8 fee recipient pubkeys, or [FEE] as fallback if not set
        """
        if cls._fee_recipients is not None:
            return cls._fee_recipients
        # Fallback to default fee if not set
        return [cls.FEE]
    
    @classmethod
    def select_random_fee_recipient(cls) -> Pubkey:
        #TODO add TokenInfo here & handle mayhem mode
        """Select a random fee recipient from the fee_recipients array (excluding main fee_recipient).
        
        This is used when building buy/sell instructions to increase transaction priority
        by randomly selecting one of the fee_recipients instead of always using the main one.
        
        The fee_recipients list should contain: [main_fee_recipient, fee_recipient_1, fee_recipient_2, ...]
        This method randomly selects from indices 1 onwards (excluding the main at index 0).
        
        Returns:
            Randomly selected fee recipient pubkey from the fee_recipients array (excluding main),
            or FEE as fallback if fee_recipients not set or insufficient
        """
        import random
        
        fee_recipients = cls.get_fee_recipients()
        if len(fee_recipients) < 2:
            # Need at least main + 1 in array to select from array
            # Fallback to default fee if fee_recipients not set or insufficient
            return cls.FEE
        
        # Main fee_recipient is at index 0, fee_recipients array starts at index 1
        # Randomly select one of the fee_recipients (indices 1 onwards)
        selected = random.choice(fee_recipients[1:])
        return selected
    
    # Account index mappings for buy/sell instructions (from IDL)
    # These are the indices where specific accounts appear in the instruction account list
    # Index 5 is user_token_account for both buy and sell instructions
    BUY_ACCOUNT_INDICES = {
        "global": 0,
        "fee": 1,
        "mint": 2,
        "bonding_curve": 3,
        "associated_bonding_curve": 4,
        "user_token_account": 5,  # This is the key one we need
        "user": 6,
        "system_program": 7,
        "token_program": 8,
        "creator_vault": 9,
        "event_authority": 10,
        "program": 11,
        "global_volume_accumulator": 12,
        "user_volume_accumulator": 13,
        "fee_config": 14,
        "fee_program": 15,
    }
    
    SELL_ACCOUNT_INDICES = {
        "global": 0,
        "fee": 1,
        "mint": 2,
        "bonding_curve": 3,
        "associated_bonding_curve": 4,
        "user_token_account": 5,  # This is the key one we need
        "user": 6,
        "system_program": 7,
        "creator_vault": 8,
        "token_program": 9,
        "event_authority": 10,
        "program": 11,
        "fee_config": 12,
        "fee_program": 13,
    }
    

    @staticmethod
    def find_global_volume_accumulator() -> Pubkey:
        """
        Derive the Program Derived Address (PDA) for the global volume accumulator.

        Returns:
            Pubkey of the derived global volume accumulator account
        """
        derived_address, _ = Pubkey.find_program_address(
            [b"global_volume_accumulator"],
            PumpFunAddresses.PROGRAM,
        )
        return derived_address

    @staticmethod
    def find_user_volume_accumulator(user: Pubkey) -> Pubkey:
        """
        Derive the Program Derived Address (PDA) for a user's volume accumulator.

        Args:
            user: Pubkey of the user account

        Returns:
            Pubkey of the derived user volume accumulator account
        """
        derived_address, _ = Pubkey.find_program_address(
            [b"user_volume_accumulator", bytes(user)],
            PumpFunAddresses.PROGRAM,
        )
        return derived_address

    @staticmethod
    def find_fee_config() -> Pubkey:
        """
        Derive the Program Derived Address (PDA) for the fee config.

        Returns:
            Pubkey of the derived fee config account
        """
        derived_address, _ = Pubkey.find_program_address(
            [b"fee_config", bytes(PumpFunAddresses.PROGRAM)],
            PumpFunAddresses.FEE_PROGRAM,
        )
        return derived_address


class PumpFunAddressProvider(AddressProvider):
    """Pump.Fun implementation of AddressProvider interface."""

    @property
    def platform(self) -> Platform:
        """Get the platform this provider serves."""
        return Platform.PUMP_FUN

    @property
    def program_id(self) -> Pubkey:
        """Get the main program ID for this platform."""
        return PumpFunAddresses.PROGRAM

    def get_system_addresses(self) -> dict[str, Pubkey]:
        """Get all system addresses required for pump.fun.

        Returns:
            Dictionary mapping address names to Pubkey objects
        """
        # Get system addresses from the single source of truth
        system_addresses = SystemAddresses.get_all_system_addresses()

        # Add pump.fun specific addresses
        pumpfun_addresses = {
            # Pump.fun specific addresses
            "program": PumpFunAddresses.PROGRAM,
            "global": PumpFunAddresses.GLOBAL,
            "event_authority": PumpFunAddresses.EVENT_AUTHORITY,
            "fee": PumpFunAddresses.FEE,
            "liquidity_migrator": PumpFunAddresses.LIQUIDITY_MIGRATOR,
            "fee_program": PumpFunAddresses.FEE_PROGRAM,
        }

        # Combine system and platform-specific addresses
        return {**system_addresses, **pumpfun_addresses}

    def derive_pool_address(
        self, base_mint: Pubkey, quote_mint: Pubkey | None = None
    ) -> Pubkey:
        """Derive the bonding curve address for a token.

        For pump.fun, this is the bonding curve PDA derived from the mint.

        Args:
            base_mint: Token mint address
            quote_mint: Not used for pump.fun (SOL is always the quote)

        Returns:
            Bonding curve address
        """
        bonding_curve, _ = Pubkey.find_program_address(
            [b"bonding-curve", bytes(base_mint)], PumpFunAddresses.PROGRAM
        )
        return bonding_curve

    def derive_user_token_account(
        self, user: Pubkey, mint: Pubkey, token_program_id: Pubkey | None = None
    ) -> Pubkey:
        """Derive user's associated token account address.

        Args:
            user: User's wallet address
            mint: Token mint address
            token_program_id: Token program (TOKEN or TOKEN_2022). Defaults to TOKEN_2022_PROGRAM

        Returns:
            User's associated token account address
        """
        if token_program_id is None:
            token_program_id = SystemAddresses.TOKEN_2022_PROGRAM
        return get_associated_token_address(user, mint, token_program_id)

    def get_additional_accounts(self, token_info: TokenInfo) -> dict[str, Pubkey]:
        """Get pump.fun-specific additional accounts needed for trading.

        Args:
            token_info: Token information

        Returns:
            Dictionary of additional account addresses
        """
        accounts = {}

        # Add bonding curve if available
        if token_info.bonding_curve:
            accounts["bonding_curve"] = token_info.bonding_curve

        # Add associated bonding curve if available
        if token_info.associated_bonding_curve:
            accounts["associated_bonding_curve"] = token_info.associated_bonding_curve

        # Add creator vault if available
        if token_info.creator_vault:
            accounts["creator_vault"] = token_info.creator_vault

        # Derive associated bonding curve if not provided
        if not token_info.associated_bonding_curve and token_info.bonding_curve:
            accounts["associated_bonding_curve"] = self.derive_associated_bonding_curve(
                token_info.mint, token_info.bonding_curve, token_info.token_program_id
            )

        # Derive creator vault if not provided but creator is available
        if not token_info.creator_vault and token_info.creator:
            accounts["creator_vault"] = self.derive_creator_vault(token_info.creator)

        return accounts

    def derive_associated_bonding_curve(
        self, mint: Pubkey, bonding_curve: Pubkey, token_program_id: Pubkey | None = None
    ) -> Pubkey:
        """Derive the associated bonding curve (ATA of bonding curve for the token).

        Args:
            mint: Token mint address
            bonding_curve: Bonding curve address
            token_program_id: Token program (TOKEN or TOKEN_2022). Defaults to TOKEN_2022_PROGRAM

        Returns:
            Associated bonding curve address
        """
        if token_program_id is None:
            token_program_id = SystemAddresses.TOKEN_2022_PROGRAM

        derived_address, _ = Pubkey.find_program_address(
            [
                bytes(bonding_curve),
                bytes(token_program_id),
                bytes(mint),
            ],
            SystemAddresses.ASSOCIATED_TOKEN_PROGRAM,
        )
        return derived_address

    def derive_creator_vault(self, creator: Pubkey) -> Pubkey:
        """Derive the creator vault address.

        Args:
            creator: Creator address

        Returns:
            Creator vault address
        """
        creator_vault, _ = Pubkey.find_program_address(
            [b"creator-vault", bytes(creator)], PumpFunAddresses.PROGRAM
        )
        return creator_vault

    def derive_global_volume_accumulator(self) -> Pubkey:
        """Derive the global volume accumulator PDA.

        Returns:
            Global volume accumulator address
        """
        return PumpFunAddresses.find_global_volume_accumulator()

    def derive_user_volume_accumulator(self, user: Pubkey) -> Pubkey:
        """Derive the user volume accumulator PDA.

        Args:
            user: User address

        Returns:
            User volume accumulator address
        """
        return PumpFunAddresses.find_user_volume_accumulator(user)

    def derive_fee_config(self) -> Pubkey:
        """Derive the fee config PDA.

        Returns:
            Fee config address
        """
        return PumpFunAddresses.find_fee_config()

    def extract_user_token_account_from_transaction(self, tx) -> Pubkey | None:
        """Extract user_token_account from transaction instruction using IDL account index.
        
        This method efficiently extracts the user_token_account from the buy/sell instruction's
        account list at the known IDL index (5 for both buy and sell) without re-parsing.
        This works even if the account was created with CreateAccountWithSeed instead of the standard ATA.
        
        Args:
            tx: Transaction object (EncodedConfirmedTransactionWithStatusMeta)
            
        Returns:
            Pubkey of user_token_account if found, None otherwise
        """
        if not tx or not tx.transaction or not tx.transaction.transaction:
            return None
        
        account_keys = tx.transaction.transaction.message.account_keys
        instructions = tx.transaction.transaction.message.instructions
        
        # Known IDL account index for user_token_account (same for buy and sell)
        user_token_account_index_in_instruction = PumpFunAddresses.BUY_ACCOUNT_INDICES.get("user_token_account")
        if user_token_account_index_in_instruction is None:
            return None
        
        # Efficiently find pump.fun instruction and extract account at index 5
        for ix in instructions:
            # Check if this is a pump.fun instruction by program_id_index
            if hasattr(ix, "program_id_index"):
                program_id_index = ix.program_id_index
                if program_id_index < len(account_keys):
                    if account_keys[program_id_index].pubkey == PumpFunAddresses.PROGRAM:
                        # Found pump.fun instruction - get accounts list directly
                        if hasattr(ix, "accounts") and ix.accounts:
                            account_indices = list(ix.accounts)
                            if user_token_account_index_in_instruction < len(account_indices):
                                transaction_account_index = account_indices[user_token_account_index_in_instruction]
                                if transaction_account_index < len(account_keys):
                                    return account_keys[transaction_account_index].pubkey
            elif hasattr(ix, "program_id"):
                # UiPartiallyDecodedInstruction - check program_id directly
                instruction_program_id = ix.program_id
                if isinstance(instruction_program_id, str):
                    instruction_program_id = Pubkey.from_string(instruction_program_id)
                elif hasattr(instruction_program_id, "pubkey"):
                    instruction_program_id = instruction_program_id.pubkey
                
                if instruction_program_id == PumpFunAddresses.PROGRAM:
                    # Found pump.fun instruction - need to find account indices
                    if hasattr(ix, "accounts") and ix.accounts:
                        # Find indices for accounts in account_keys
                        account_indices = []
                        for acc in ix.accounts:
                            acc_pubkey = acc
                            if isinstance(acc, str):
                                acc_pubkey = Pubkey.from_string(acc)
                            elif hasattr(acc, "pubkey"):
                                acc_pubkey = acc.pubkey
                            
                            # Find index in account_keys
                            for i, key in enumerate(account_keys):
                                if key.pubkey == acc_pubkey:
                                    account_indices.append(i)
                                    break
                        
                        if user_token_account_index_in_instruction < len(account_indices):
                            transaction_account_index = account_indices[user_token_account_index_in_instruction]
                            if transaction_account_index < len(account_keys):
                                return account_keys[transaction_account_index].pubkey
        
        return None

    def get_fee_recipient(self, token_info: TokenInfo) -> Pubkey:
        """Get the correct fee recipient based on mayhem mode.

        Args:
            token_info: Token information with is_mayhem_mode flag

        Returns:
            Fee recipient address (mayhem or standard)
        """
        #TODO Move this back into the code
        if token_info.is_mayhem_mode:
            return PumpFunAddresses.MAYHEM_FEE
        return PumpFunAddresses.FEE

    def get_buy_instruction_accounts(
        self, token_info: TokenInfo, user: Pubkey
    ) -> dict[str, Pubkey]:
        """Get all accounts needed for a buy instruction.

        Args:
            token_info: Token information
            user: User's wallet address

        Returns:
            Dictionary of account addresses for buy instruction
        Note:
            user_token_account is NOT included here - it must be extracted from the actual
            transaction instruction using extract_user_token_account_from_transaction() since
            it may be created with CreateAccountWithSeed instead of standard ATA.
        """
        additional_accounts = self.get_additional_accounts(token_info)

        # Determine token program to use
        token_program_id = (
            token_info.token_program_id
            if token_info.token_program_id
            else SystemAddresses.TOKEN_PROGRAM
        )

        # Determine fee recipient based on mayhem mode
        fee_recipient = self.get_fee_recipient(token_info)

        return {
            "global": PumpFunAddresses.GLOBAL,
            "fee": fee_recipient,
            "mint": token_info.mint,
            "bonding_curve": additional_accounts.get(
                "bonding_curve", token_info.bonding_curve
            ),
            "associated_bonding_curve": additional_accounts.get(
                "associated_bonding_curve", token_info.associated_bonding_curve
            ),
            # user_token_account is NOT included - must be extracted from transaction
            "user": user,
            "system_program": SystemAddresses.SYSTEM_PROGRAM,
            "token_program": token_program_id,
            "creator_vault": additional_accounts.get(
                "creator_vault", token_info.creator_vault
            ),
            "event_authority": PumpFunAddresses.EVENT_AUTHORITY,
            "program": PumpFunAddresses.PROGRAM,
            "global_volume_accumulator": self.derive_global_volume_accumulator(),
            "user_volume_accumulator": self.derive_user_volume_accumulator(user),
            "fee_config": self.derive_fee_config(),
            "fee_program": PumpFunAddresses.FEE_PROGRAM,
        }

    def get_sell_instruction_accounts(
        self, token_info: TokenInfo, user: Pubkey
    ) -> dict[str, Pubkey]:
        """Get all accounts needed for a sell instruction.

        Args:
            token_info: Token information
            user: User's wallet address

        Returns:
            Dictionary of account addresses for sell instruction
        Note:
            user_token_account is NOT included here - it must be extracted from the actual
            transaction instruction using extract_user_token_account_from_transaction() since
            it may be created with CreateAccountWithSeed instead of standard ATA.
        """
        additional_accounts = self.get_additional_accounts(token_info)

        # Determine token program to use
        token_program_id = (
            token_info.token_program_id
            if token_info.token_program_id
            else SystemAddresses.TOKEN_PROGRAM
        )

        # Determine fee recipient based on mayhem mode
        fee_recipient = self.get_fee_recipient(token_info)

        return {
            "global": PumpFunAddresses.GLOBAL,
            "fee": fee_recipient,
            "mint": token_info.mint,
            "bonding_curve": additional_accounts.get(
                "bonding_curve", token_info.bonding_curve
            ),
            "associated_bonding_curve": additional_accounts.get(
                "associated_bonding_curve", token_info.associated_bonding_curve
            ),
            # user_token_account is NOT included - must be extracted from transaction
            "user": user,
            "system_program": SystemAddresses.SYSTEM_PROGRAM,
            "creator_vault": additional_accounts.get(
                "creator_vault", token_info.creator_vault
            ),
            "token_program": token_program_id,
            "event_authority": PumpFunAddresses.EVENT_AUTHORITY,
            "program": PumpFunAddresses.PROGRAM,
            "fee_config": self.derive_fee_config(),
            "fee_program": PumpFunAddresses.FEE_PROGRAM,
        }
