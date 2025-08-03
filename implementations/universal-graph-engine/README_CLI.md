# Universal Graph Engine - Command Line Interface

The Universal Graph Engine CLI provides an interactive command-line interface for creating, manipulating, and visualizing complex universal graphs with maximum ease of use.

## Features

### Core Operations
- **Easy Node Creation**: Create nodes with various data types
- **Simple Edge Management**: Connect nodes with typed relationships
- **Advanced Graph Manipulation**: Insert nodes between edges, create triangular connections
- **Interactive Shell**: Full-featured command shell with help system
- **Batch Processing**: Execute script files and process commands from stdin

### Advanced Features
- **Insert Node Between Edges**: Seamlessly insert intermediate nodes in existing connections
- **Triangle Creation**: Create nodes that connect to multiple existing nodes
- **Hyperedges**: Support for N-ary relationships connecting multiple nodes
- **Property Management**: Set and get properties on nodes and edges
- **Graph Analysis**: Statistics, neighbor discovery, and path finding

## Quick Start

### Building the CLI

```bash
mkdir build && cd build
cmake .. -DUG_BUILD_CLI=ON
make ug_cli
```

### Interactive Mode

```bash
./ug_cli
```

This starts the interactive shell where you can type commands directly.

### Script Mode

```bash
./ug_cli -f script.ug
```

Execute commands from a script file.

### Batch Mode

```bash
./ug_cli --batch < commands.txt
```

Process commands from stdin in batch mode.

## Command Reference

### Basic Node and Edge Operations

#### Create Node
```bash
node <name> [data] [type]
```

Examples:
```bash
node alice "Alice Smith"
node count 42 int
node temp 3.14 float
```

#### Create Edge
```bash
edge <from> <to> [type] [weight]
```

Examples:
```bash
edge alice bob KNOWS 1.0
edge node1 node2 CONNECTED
```

### Advanced Operations

#### Insert Node Between Edges
```bash
insert <from> <to> <new_node_name> [data]
```

This command:
1. Creates a new node with the specified name and data
2. Removes any direct edge between `from` and `to`
3. Creates edges: `from` → `new_node` → `to`

Examples:
```bash
insert alice bob charlie "Charlie Brown"
insert n1 n2 middle
```

#### Create Triangle Connection
```bash
triangle <node1> <node2> <new_node> [data]
```

This command:
1. Creates a new node
2. Connects the new node to both existing nodes
3. Forms a triangular relationship pattern

Examples:
```bash
triangle alice bob charlie "Charlie connects both"
triangle n1 n2 connector
```

#### Create Hyperedge
```bash
hyperedge <type> <node1> <node2> [node3] ...
```

Examples:
```bash
hyperedge MEETING alice bob charlie
hyperedge COLLABORATION n1 n2 n3 n4
```

### Information and Analysis

#### List Nodes
```bash
nodes [pattern]
```

#### List Edges
```bash
edges [from_pattern] [to_pattern]
```

#### Show Node Details
```bash
show <name>
```

#### Find Neighbors
```bash
neighbors <name> [depth]
```

#### Find Path
```bash
path <from> <to> [max_depth]
```

#### Graph Statistics
```bash
stats
```

### Property Management

#### Set Property
```bash
set <node> <property> <value> [type]
```

Examples:
```bash
set alice age 25 int
set bob city "New York"
```

#### Get Property
```bash
get <node> <property>
```

### File Operations

#### Save Graph
```bash
save <filename>
```

#### Load Graph
```bash
load <filename>
```

#### Export Graph
```bash
export <format> <filename>
```

Supported formats: `dot`, `json`, `xml`, `csv`

#### Import Graph
```bash
import <format> <filename>
```

### Utility Commands

#### Help
```bash
help [command]
```

Show general help or help for specific command.

#### Clear Graph
```bash
clear
```

Remove all nodes and edges from the current graph.

#### Visualize
```bash
viz [layout]
```

Display ASCII visualization of the graph.

#### Quit
```bash
quit
exit
q
```

## Example Session

```bash
$ ./ug_cli
Universal Graph Engine CLI v1.0
Type 'help' for available commands, 'quit' to exit.

ug> node alice "Alice Smith"
SUCCESS: Created node 'alice' with ID 1

ug> node bob "Bob Johnson"
SUCCESS: Created node 'bob' with ID 2

ug> edge alice bob KNOWS 0.8
SUCCESS: Created edge alice -> bob (type: KNOWS, weight: 0.80, ID: 1)

ug> insert alice bob charlie "Charlie Brown"
SUCCESS: Inserted node 'charlie' between 'alice' and 'bob'

ug> triangle alice charlie david "David Wilson"
SUCCESS: Created triangle: 'david' connects to both 'alice' and 'charlie'

ug> nodes
Nodes in graph (4 total):
Name                 ID         Data
----                 --         ----
alice                1          <data>
bob                  2          <data>
charlie              3          <data>
david                4          <data>

ug> stats
Graph Statistics:
  Nodes: 4
  Relationships: 4
  Named nodes: 4
  Graph density: 0.6667

ug> quit
Goodbye!
```

## Script Files

Create script files with `.ug` extension:

```bash
# demo.ug - Graph creation script

# Create nodes
node alice "Alice Smith"
node bob "Bob Johnson"
node charlie "Charlie Brown"

# Create connections
edge alice bob KNOWS 0.8
edge bob charlie FRIENDS 0.9

# Insert intermediate node
insert alice bob david "David Wilson"

# Create triangle
triangle charlie bob eve "Eve connects both"

# Show results
stats
nodes
```

Run with:
```bash
./ug_cli -f demo.ug
```

## Advanced Use Cases

### Social Network Modeling
```bash
# Create people
node alice "Alice Smith"
node bob "Bob Johnson"
node charlie "Charlie Brown"
node diana "Diana Prince"

# Create relationships
edge alice bob FRIENDS 0.9
edge bob charlie COLLEAGUES 0.7
edge charlie diana FAMILY 1.0

# Insert mediator
insert alice diana eve "Eve (mutual friend)"

# Create group connection
triangle alice bob group "Study Group"
```

### Knowledge Graph Construction
```bash
# Create concepts
node probability "Probability Theory"
node statistics "Statistics"
node ml "Machine Learning"

# Create relationships
edge probability statistics FOUNDATION_OF 1.0
edge statistics ml ENABLES 0.8

# Insert bridging concept
insert probability ml bayes "Bayesian Methods"

# Create interdisciplinary connection
triangle statistics ml data_science "Data Science"
```

### Workflow Modeling
```bash
# Create process steps
node start "Start Process"
node validate "Validate Input"
node process "Process Data"
node end "End Process"

# Create workflow
edge start validate NEXT 1.0
edge validate process NEXT 1.0
edge process end NEXT 1.0

# Insert error handling
insert validate process error_check "Error Checking"

# Add monitoring
triangle process end monitor "Process Monitor"
```

## Building and Installation

### Prerequisites
- CMake 3.15+
- C99 compatible compiler
- Universal Graph Engine core library

### Build Options
```bash
cmake .. \
  -DUG_BUILD_CLI=ON \
  -DUG_BUILD_SHARED=ON \
  -DUG_BUILD_STATIC=ON
```

### Installation
```bash
make install
```

This installs:
- `ug_cli` executable to `${CMAKE_INSTALL_BINDIR}`
- CLI library headers (if needed for extensions)

## Extending the CLI

The CLI is designed to be extensible. You can:

1. **Add New Commands**: Implement command handlers in `ug_cli.c`
2. **Custom Data Types**: Extend the value parsing system
3. **Visualization Plugins**: Add new graph layout algorithms
4. **Export Formats**: Implement additional file format support

## Performance Notes

- The CLI is optimized for interactive use with moderate-sized graphs
- For large graphs (>10K nodes), consider using batch mode
- Memory usage scales linearly with graph size
- Command parsing is lightweight and fast

## Troubleshooting

### Common Issues

**Command not recognized**
```bash
ug> unknown_command
ERROR: Unknown command: unknown_command. Type 'help' for available commands.
```

**Node not found**
```bash
ug> edge alice nonexistent KNOWS
ERROR: Node 'nonexistent' not found
```

**Script file errors**
```bash
$ ./ug_cli -f missing.ug
ERROR: Cannot open script file 'missing.ug'
```

### Debug Mode

Enable verbose output:
```bash
./ug_cli --verbose
```

### Getting Help

1. Use `help` command for general information
2. Use `help <command>` for specific command help
3. Check the examples in the `examples/` directory
4. Refer to the Universal Graph Engine documentation

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.