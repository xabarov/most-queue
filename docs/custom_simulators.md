# Guide to Building Custom Simulators

[🇷🇺 Русская версия](custom_simulators.ru.md)

This guide describes how to create your own queueing system simulator classes using the new event-based architecture.

## Architecture Overview

The simulators now use an event-based approach with an event system and hook methods for extending functionality. This makes creating custom simulators significantly easier.

### Core Components

1. **BaseSimulationCore** - base class with shared functionality
2. **QsSim** - main simulator class with event-based logic
3. **EventScheduler** - event manager for scheduling
4. **Hook methods** - extension points for customization

## Basic Example

Here is a minimal example of a custom simulator:

```python
from most_queue.sim.base import QsSim

class MyCustomSimulator(QsSim):
    def __init__(self, num_of_channels, buffer=None, verbose=True):
        super().__init__(num_of_channels, buffer, verbose)
        # Your initialization
    
    def _get_custom_events(self):
        """Register custom events"""
        return {
            'my_event': self.my_event_time  # dict: event_type -> time
        }
    
    def _handle_custom_event(self, event_type):
        """Handle custom events"""
        if event_type == 'my_event':
            self.handle_my_event()
        else:
            super()._handle_custom_event(event_type)
    
    def handle_my_event(self):
        """Handling logic for your event"""
        # Your logic here
        pass
```

## Hook Methods

### Event Registration Methods

#### `_get_custom_events()`

Returns a dictionary of custom events. The key is the event type (a string), the value is the event time.

```python
def _get_custom_events(self):
    events = {}
    if self.some_condition:
        events['special_event'] = self.special_event_time
    return events
```

#### `_handle_custom_event(event_type)`

Handles custom events. Must be overridden to handle your events.

```python
def _handle_custom_event(self, event_type):
    if event_type == 'my_event':
        # handling
        pass
    else:
        super()._handle_custom_event(event_type)  # for unknown events
```

### Methods for Extending Base Behavior

#### `_before_arrival()` / `_after_arrival()`

Called before and after processing a job arrival event.

```python
def _before_arrival(self, moment=None, ts=None):
    # Logic before arrival
    pass

def _after_arrival(self, moment=None, ts=None):
    # Logic after arrival
    pass
```

#### `_before_serving()` / `_after_serving()`

Called before and after a service completion.

```python
def _before_serving(self, channel, is_network=False):
    # Logic before serving
    pass

def _after_serving(self, channel, task=None, is_network=False):
    # Logic after serving
    pass
```

### Helper Methods

#### `_update_state_probs(old_time, new_time, state)`

Updates the array of system state probabilities. Use this method instead of modifying `self.p` directly.

```python
# Correct:
self._update_state_probs(self.ttek, new_time, self.in_sys)

# Incorrect:
self.p[self.in_sys] += new_time - self.ttek
```

#### `_mark_servers_time_changed()`

Marks that the server times have changed. Call it after modifying `time_to_end_service`.

```python
self.servers[0].time_to_end_service = new_time
self._mark_servers_time_changed()
```

## Usage Examples

### Example 1: Simulator with an Additional Event

A simulator with a periodic equipment maintenance event:

```python
from most_queue.sim.base import QsSim

class MaintenanceSimulator(QsSim):
    def __init__(self, num_of_channels, maintenance_interval, buffer=None):
        super().__init__(num_of_channels, buffer)
        self.maintenance_interval = maintenance_interval
        self.next_maintenance_time = maintenance_interval
    
    def _get_custom_events(self):
        return {'maintenance': self.next_maintenance_time}
    
    def _handle_custom_event(self, event_type):
        if event_type == 'maintenance':
            # Take all servers down for maintenance
            for server in self.servers:
                if not server.is_free:
                    # Interrupt service
                    server.end_service()
                    self.free_channels += 1
                    self._free_servers.add(server.id - 1)
            
            # Schedule the next maintenance
            self.next_maintenance_time = self.ttek + self.maintenance_interval
            self._mark_servers_time_changed()
        else:
            super()._handle_custom_event(event_type)
```

### Example 2: Simulator with Priority Adjustment

A simulator that changes job priorities depending on their waiting time:

```python
from most_queue.sim.base import QsSim
from most_queue.sim.utils.tasks import Task

class DynamicPrioritySimulator(QsSim):
    def __init__(self, num_of_channels, priority_threshold, buffer=None):
        super().__init__(num_of_channels, buffer)
        self.priority_threshold = priority_threshold
    
    def _after_arrival(self, moment=None, ts=None):
        # After an arrival, check the waiting time in the queue
        # and promote old jobs
        for task in self.queue.queue:
            wait_time = self.ttek - task.start_waiting_time
            if wait_time > self.priority_threshold:
                # Move to the front of the queue (raise the priority)
                self.queue.queue.remove(task)
                self.queue.queue.appendleft(task)
                break
```

### Example 3: Extending an Existing Simulator

Adding functionality to an existing simulator:

```python
from most_queue.sim.impatient import ImpatientQueueSim

class EnhancedImpatientSimulator(ImpatientQueueSim):
    def __init__(self, num_of_channels, buffer=None):
        super().__init__(num_of_channels, buffer)
        self.reneged_count = 0  # counter of reneged jobs
    
    def _handle_custom_event(self, event_type):
        if event_type == 'task_drop':
            self.reneged_count += 1
            # Invoke the base handling
            super()._handle_custom_event(event_type)
        else:
            super()._handle_custom_event(event_type)
```

## Overriding Standard Events

If you need to change the handling logic of standard events (arrival, serving), you can override the `_get_available_events()` and `_execute_event()` methods:

```python
def _get_available_events(self):
    """Override to add extra logic"""
    events = super()._get_available_events()
    
    # Add filtering or additional events
    if self.some_condition:
        events['special_arrival'] = self.special_arrival_time
    
    return events

def _execute_event(self, event_type):
    """Override to change the execution logic"""
    if event_type == 'special_arrival':
        self.handle_special_arrival()
    else:
        super()._execute_event(event_type)
```

## Patterns for Typical Extensions

### Adding a New Job Arrival Type

```python
def _get_available_events(self):
    events = super()._get_available_events()
    events['vip_arrival'] = self.vip_arrival_time
    return events

def _execute_event(self, event_type):
    if event_type == 'vip_arrival':
        self.handle_vip_arrival()
    else:
        super()._execute_event(event_type)
```

### Adding Periodic Events

```python
def __init__(self, ...):
    super().__init__(...)
    self.periodic_event_interval = 100.0
    self.next_periodic_event = self.periodic_event_interval

def _get_custom_events(self):
    return {'periodic': self.next_periodic_event}

def _handle_custom_event(self, event_type):
    if event_type == 'periodic':
        self.handle_periodic_event()
        self.next_periodic_event = self.ttek + self.periodic_event_interval
```

### Adding Conditional Events

```python
def _get_custom_events(self):
    events = {}
    # The event only fires under certain conditions
    if self.in_sys > 10:  # system overload
        events['overload_alert'] = self.ttek  # immediately
    return events
```

## Tips and Best Practices

1. **Always call `super()`** in overridden methods unless you are completely replacing the functionality
2. **Use `_update_state_probs()`** to update the state probabilities
3. **Call `_mark_servers_time_changed()`** after changing server times
4. **Document custom events** - use clear names and describe their purpose
5. **Test in isolation** - make sure your changes do not break the base functionality

## Migrating Existing Simulators

If you have an existing simulator that overrides `run_one_step()`, you can migrate it as follows:

**Before:**
```python
def run_one_step(self):
    # Complex logic for choosing the next event
    if condition1:
        self.handle_event1()
    elif condition2:
        self.handle_event2()
    # ...
```

**After:**
```python
def _get_custom_events(self):
    events = {}
    if condition1:
        events['event1'] = self.event1_time
    if condition2:
        events['event2'] = self.event2_time
    return events

def _handle_custom_event(self, event_type):
    if event_type == 'event1':
        self.handle_event1()
    elif event_type == 'event2':
        self.handle_event2()
    # run_one_step() is now inherited and uses the events automatically
```

## Building Custom Queueing Networks

Queueing networks also support the event-based architecture. The principles are the same as for standalone simulators, but there are specifics related to working with network nodes.

### Basic Custom Network Example

```python
from most_queue.sim.networks.network import NetworkSimulator

class MyCustomNetwork(NetworkSimulator):
    def __init__(self):
        super().__init__()
        # Your initialization
    
    def _get_custom_network_events(self):
        """Register custom network events"""
        return {
            'network_maintenance': self.next_maintenance_time
        }
    
    def _handle_custom_network_event(self, event_type):
        """Handle custom network events"""
        if event_type == 'network_maintenance':
            self.perform_maintenance()
        else:
            super()._handle_custom_network_event(event_type)
```

### Hook Methods for Networks

#### `_before_network_arrival(k=None)` / `_after_network_arrival(k=None)`

Called before and after an external job arrival to the network.

```python
def _before_network_arrival(self, k=None):
    # k - class index for PriorityNetwork, None for regular networks
    # Logic before arrival
    pass
```

#### `_before_node_serving(node, channel)` / `_after_node_serving(node, channel, task)`

Called before and after service at a specific network node.

```python
def _before_node_serving(self, node, channel):
    # node - node index
    # channel - channel index
    # Logic before serving
    pass
```

### Example: Network with Periodic Node Maintenance

```python
from most_queue.sim.networks.network import NetworkSimulator

class MaintenanceNetwork(NetworkSimulator):
    """
    Network with periodic maintenance of all nodes.
    """
    def __init__(self, maintenance_interval):
        super().__init__()
        self.maintenance_interval = maintenance_interval
        self.next_maintenance_time = maintenance_interval
        self.maintenance_count = 0
    
    def _get_custom_network_events(self):
        """Register the maintenance event"""
        return {'network_maintenance': self.next_maintenance_time}
    
    def _handle_custom_network_event(self, event_type):
        """Handle maintenance"""
        if event_type == 'network_maintenance':
            self.maintenance_count += 1
            # Take all nodes down for maintenance
            for node_idx, node in enumerate(self.qs):
                # Node maintenance logic
                pass
            # Schedule the next maintenance
            self.next_maintenance_time = self.ttek + self.maintenance_interval
        else:
            super()._handle_custom_network_event(event_type)
    
    def _before_node_serving(self, node, channel):
        """Check whether network maintenance is in progress"""
        # You can add checks before serving here
        pass
```

### Working with Node Events

Service events at the nodes are collected automatically by the `_get_node_serving_events()` method. Event format: `'node_serving_{node}_{channel}'`.

You can override `_get_node_serving_events()` to customize event collection:

```python
def _get_node_serving_events(self):
    """Override to filter or modify events"""
    events = super()._get_node_serving_events()
    # Add filtering or modification
    return events
```

## Additional Resources

- See the examples in `examples/custom_simulator_example.py`
- Study the existing simulators (`vacations.py`, `impatient.py`, `negative.py`) as usage examples
- Study the networks (`networks/network.py`, `networks/priority_network.py`, `networks/negative_network.py`) for network examples
- Base class documentation: `most_queue/sim/base.py` and `most_queue/sim/networks/base_network_sim.py`
