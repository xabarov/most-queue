"""
Queueing-inventory systems: queues where service consumes a unit of stock,
replenished with random lead time.
"""

from most_queue.theory.inventory.mm1_inventory import MM1QueueingInventoryCalc
from most_queue.theory.inventory.mmc_inventory import MMcQueueingInventoryCalc

__all__ = [
    "MM1QueueingInventoryCalc",
    "MMcQueueingInventoryCalc",
]
