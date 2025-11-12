-- Database migration script to add signature column to pumpportal_messages table.
-- This migration:
--   1. Adds signature column to store the transaction signature from PumpPortal messages
--
-- Usage: sqlite3 <database_path> < migrate_add_signature.sql

-- Add signature column
ALTER TABLE pumpportal_messages ADD COLUMN signature TEXT;

