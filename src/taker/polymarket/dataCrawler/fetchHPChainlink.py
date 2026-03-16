# THIS IS EXAMPLE CODE THAT USES HARDCODED VALUES FOR CLARITY.
# THIS IS EXAMPLE CODE THAT USES UN-AUDITED CODE.
# DEVELOPMENT ONLY. DO NOT USE THIS CODE IN PRODUCTION.

import os
import json
from dotenv import load_dotenv
from web3 import Web3

# Load environment variables from .env file
load_dotenv()

# Determine RPC endpoint
rpc_url = os.getenv('INFURA_RPC_URL')
if not rpc_url:
    raise ValueError("INFURA_RPC_URL not found in .env file")

web3 = Web3(Web3.HTTPProvider(rpc_url))

# AggregatorV3Interface ABI - https://docs.chain.link/data-feeds/developer-reference#abi
abi_string = '[{"inputs":[],"name":"decimals","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"description","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint80","name":"_roundId","type":"uint80"}],"name":"getRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"latestRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"version","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]'

# Feed address - https://docs.chain.link/data-feeds/price-feeds/addresses?network=polygon&search=bt&page=1
# The BTC/USD price feed Contract on Chainlink
addr = '0xc907E116054Ad103354f2D350FD2514433D57F6f'

# Set up contract instance
contract = web3.eth.contract(address=addr, abi=json.loads(abi_string))

# Valid roundId must be known. They are NOT timestamp.
# 1 Round may contains different interval
# Scenario	Result
# Price drops 5% in 1 second	✅ New round created (deviation triggered)
# Price stable for 30 minutes	❌ No new round yet (within deviation)
# Price stable for 1 hour	✅ New round created (heartbeat triggered)
# Reference: https://docs.chain.link/data-feeds/historical-data

def get_latest_round_data():
    """
    Fetch the latest round data and extract roundId, price, and timestamp.
    Returns: (roundId, price, timestamp, decimals)
    """
    round_id, answer, _, updated_at, _ = contract.functions.latestRoundData().call()
    decimals = contract.functions.decimals().call()
    return round_id, answer, updated_at, decimals


def decode_round_id(round_id):
    """
    Decode roundId into phaseId and aggregatorRoundId.
    Formula: roundId = (phaseId << 64) | aggregatorRoundId
    """
    phase_id = round_id >> 64
    aggregator_round_id = round_id & 0xFFFFFFFFFFFFFFFF
    return phase_id, aggregator_round_id


def get_historical_data(start_round_id, num_rounds=10):
    """
    Fetch historical price data for a range of roundIds.
    Starts from start_round_id and goes backwards.
    
    Args:
        start_round_id: The roundId to start from (usually from latestRoundData)
        num_rounds: Number of historical rounds to fetch
    
    Returns:
        List of tuples: (roundId, price, timestamp)
    """
    historical_data = []
    
    for i in range(num_rounds):
        round_id = start_round_id - i
        if round_id < 0:
            break
        
        try:
            _, answer, _, updated_at, _ = contract.functions.getRoundData(round_id).call()
            # Only include rounds where timestamp is not zero (valid rounds)
            if updated_at > 0:
                historical_data.append({
                    'roundId': round_id,
                    'price': answer,
                    'timestamp': updated_at
                })
        except Exception as e:
            # Invalid roundId or RPC error - continue to next
            continue
    
    return historical_data


# Example usage:
if __name__ == "__main__":
    # Get latest round data
    print("=== Latest Round Data ===")
    latest_round_id, latest_price, latest_timestamp, decimals = get_latest_round_data()
    print(f"Round ID: {latest_round_id}")
    print(f"Price: {latest_price / (10 ** decimals)}")
    print(f"Timestamp: {latest_timestamp}")
    print(f"Decimals: {decimals}\n")
    
    # Decode roundId
    print("=== Decode Round ID ===")
    phase_id, agg_round_id = decode_round_id(latest_round_id)
    print(f"Phase ID: {phase_id}")
    print(f"Aggregator Round ID: {agg_round_id}\n")
    
    # Get historical data
    print("=== Historical Data (last 10 rounds) ===")
    history = get_historical_data(latest_round_id, num_rounds=10)
    for data in history:
        print(f"Round {data['roundId']}: ${data['price'] / (10 ** decimals)} (ts: {data['timestamp']})")
