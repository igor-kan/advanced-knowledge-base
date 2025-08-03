# Universal Graph Engine - Complete Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [C Programming Basics](#c-programming-basics)
4. [Graph Theory Fundamentals](#graph-theory-fundamentals)
5. [Installation Guide](#installation-guide)
6. [Command Line Interface](#command-line-interface)
7. [Programming API](#programming-api)
8. [Advanced Features](#advanced-features)
9. [Examples and Tutorials](#examples-and-tutorials)
10. [Language Migration Guide](#language-migration-guide)
11. [Troubleshooting](#troubleshooting)
12. [FAQ](#faq)

---

## Introduction

### What is the Universal Graph Engine?

The Universal Graph Engine is the most advanced and flexible graph database system ever created. Think of it as a super-powered way to store and work with connected information - like a social network, a family tree, or a map of how different concepts relate to each other.

### Why "Universal"?

- **Any Data Type**: Store text, numbers, images, or any custom data
- **Any Relationship**: Model simple connections or complex multi-way relationships
- **Any Scale**: From small personal projects to massive enterprise systems
- **Any Language**: Start in C, migrate to C++, Rust, Go, Python, or others

### What Makes It Special?

Unlike traditional databases that store information in tables (like spreadsheets), graph databases store information as **nodes** (things) connected by **edges** (relationships). This makes it incredibly powerful for understanding how things relate to each other.

**Example**: In a social network
- **Nodes**: People (Alice, Bob, Charlie)
- **Edges**: Relationships (Alice KNOWS Bob, Bob FRIENDS Charlie)

### Who Should Use This?

- **Beginners**: Learning programming and want to understand data structures
- **Students**: Working on computer science projects or research
- **Developers**: Building applications that need to track relationships
- **Researchers**: Analyzing complex networks and connections
- **Enterprises**: Managing large-scale connected data systems

---

## Getting Started

### Prerequisites

Before you begin, you'll need:

1. **A Computer** running Windows, macOS, or Linux
2. **Basic Command Line Knowledge** (we'll teach you what you need)
3. **A Text Editor** (Notepad++, VS Code, or even plain Notepad)
4. **Curiosity and Patience** (programming takes practice!)

### What You'll Learn

By the end of this documentation, you'll be able to:
- Create and manipulate complex graph structures
- Write programs that use the Universal Graph Engine
- Understand fundamental programming concepts
- Build real-world applications with connected data

---

## C Programming Basics

### What is C?

C is a programming language created in the 1970s that's still widely used today. It's like learning Latin for programming - many other languages are based on C, so learning it helps you understand programming in general.

### Why Start with C?

1. **Fundamental**: Teaches you how computers really work
2. **Fast**: C programs run very quickly
3. **Portable**: C code works on almost any computer
4. **Foundation**: Makes learning other languages easier

### Basic C Concepts

#### Variables: Storing Information

Think of variables like labeled boxes that hold information:

```c
// This creates a box labeled 'age' that holds the number 25
int age = 25;

// This creates a box labeled 'name' that holds text
char name[] = "Alice";

// This creates a box that holds decimal numbers
float temperature = 98.6;
```

**Types of Variables:**
- `int`: Whole numbers (1, 42, -5)
- `float/double`: Decimal numbers (3.14, -2.5)
- `char`: Single characters ('A', 'x', '!')
- `char[]`: Text strings ("Hello", "Alice Smith")

#### Functions: Reusable Instructions

Functions are like recipes - you write them once and use them many times:

```c
// This function adds two numbers together
int add_numbers(int first, int second) {
    int result = first + second;
    return result;
}

// This function prints a greeting
void say_hello(char name[]) {
    printf("Hello, %s!\n", name);
}
```

**Function Parts:**
- `int add_numbers` - Function name and what it returns
- `(int first, int second)` - Inputs (parameters)
- `{ ... }` - The instructions (function body)
- `return result` - What the function gives back

#### Pointers: Advanced Variable References

Pointers are like addresses - instead of holding data directly, they point to where data is stored:

```c
int age = 25;        // A variable holding 25
int* age_pointer;    // A pointer (like an address)
age_pointer = &age;  // Point to where 'age' is stored

// Now you can access age through the pointer
printf("Age is: %d\n", *age_pointer);  // Prints: Age is: 25
```

**Why Use Pointers?**
- **Efficiency**: Don't copy large data, just point to it
- **Flexibility**: Change what you're pointing to
- **Memory Management**: Control exactly how memory is used

#### Structures: Grouping Related Data

Structures let you group related information together:

```c
// Define a structure for a person
struct Person {
    char name[50];
    int age;
    float height;
};

// Create a person
struct Person alice;
strcpy(alice.name, "Alice Smith");
alice.age = 28;
alice.height = 5.6;
```

#### Dynamic Memory: Creating Data at Runtime

Sometimes you don't know how much data you'll need until your program is running:

```c
// Ask for memory to store 10 integers
int* numbers = malloc(10 * sizeof(int));

// Use the memory
numbers[0] = 42;
numbers[1] = 100;

// Always free memory when done!
free(numbers);
```

### Your First C Program

Let's write a simple program that creates and uses a graph:

```c
#include <stdio.h>
#include "universal_graph.h"

int main() {
    // Create a new graph
    ug_graph_t* my_graph = ug_create_graph();
    
    // Create some nodes
    ug_node_id_t alice = ug_create_string_node(my_graph, "Alice");
    ug_node_id_t bob = ug_create_string_node(my_graph, "Bob");
    
    // Connect them
    ug_create_edge(my_graph, alice, bob, "KNOWS", 1.0);
    
    // Show what we created
    ug_print_graph_stats(my_graph);
    
    // Clean up
    ug_destroy_graph(my_graph);
    
    return 0;
}
```

**What This Program Does:**
1. **Line 1-2**: Include necessary files
2. **Line 4**: Start of our main function
3. **Line 5**: Create a new, empty graph
4. **Lines 7-8**: Create two nodes with names
5. **Line 11**: Create an edge connecting the nodes
6. **Line 14**: Print information about our graph
7. **Line 17**: Clean up memory
8. **Line 19**: End the program successfully

---

## Graph Theory Fundamentals

### What is a Graph?

In mathematics and computer science, a **graph** is a collection of things (called **nodes** or **vertices**) connected by relationships (called **edges** or **links**).

**Important**: This is NOT the same as a bar chart or line graph you might see in Excel!

### Basic Graph Concepts

#### Nodes (Vertices)
Nodes represent entities - things that exist:
- **People**: Alice, Bob, Charlie
- **Places**: New York, London, Tokyo
- **Concepts**: Love, Mathematics, Programming
- **Objects**: Car, House, Computer

#### Edges (Links/Relationships)
Edges represent relationships between nodes:
- **Social**: Alice KNOWS Bob
- **Geographical**: New York CONNECTED_TO London
- **Conceptual**: Mathematics ENABLES Programming
- **Ownership**: Alice OWNS Car

### Types of Graphs

#### Directed vs Undirected

**Undirected Graph**: Relationships go both ways
```
Alice ←→ Bob (Alice knows Bob AND Bob knows Alice)
```

**Directed Graph**: Relationships have direction
```
Alice → Bob (Alice knows Bob, but maybe Bob doesn't know Alice)
```

#### Weighted vs Unweighted

**Unweighted**: All relationships are equal
```
Alice — Bob — Charlie (simple connections)
```

**Weighted**: Relationships have strength/importance
```
Alice —0.9— Bob —0.3— Charlie (Alice and Bob are close friends, Bob barely knows Charlie)
```

### Advanced Graph Concepts

#### Hypergraphs: Beyond Simple Connections

Traditional graphs connect two nodes at a time:
```
Alice — Bob — Charlie
```

Hypergraphs can connect multiple nodes simultaneously:
```
Meeting(Alice, Bob, Charlie, Diana) - all four people in one meeting
```

#### Temporal Graphs: Time-Aware Connections

Relationships can change over time:
```
2020: Alice WORKS_WITH Bob
2021: Alice MANAGES Bob  
2022: Alice COLLABORATES_WITH Bob
```

#### Meta-Relationships: Relationships Between Relationships

You can have relationships about relationships:
```
Alice KNOWS Bob (relationship 1)
Charlie KNOWS Diana (relationship 2)
Relationship1 SIMILAR_TO Relationship2
```

### Real-World Graph Examples

#### Social Networks
- **Nodes**: People, Groups, Pages
- **Edges**: Friendships, Likes, Shares, Comments
- **Use Cases**: Friend suggestions, content recommendations

#### Transportation Networks
- **Nodes**: Cities, Airports, Train Stations
- **Edges**: Routes, with weights representing distance or time
- **Use Cases**: Finding shortest routes, optimizing schedules

#### Knowledge Graphs
- **Nodes**: Concepts, Facts, Entities
- **Edges**: Relationships between concepts
- **Use Cases**: Search engines, AI reasoning, education

#### Computer Networks
- **Nodes**: Computers, Routers, Servers
- **Edges**: Network connections with bandwidth weights
- **Use Cases**: Network optimization, failure detection

---

## Installation Guide

### System Requirements

**Minimum Requirements:**
- Operating System: Windows 7+, macOS 10.12+, or Linux (any modern distribution)
- RAM: 512 MB available
- Disk Space: 100 MB for basic installation
- Processor: Any modern CPU (x86_64 or ARM)

**Recommended Requirements:**
- RAM: 2 GB+ for large graphs
- Disk Space: 1 GB+ for development
- SSD storage for better performance

### Step 1: Install Development Tools

#### On Windows

1. **Install Visual Studio Community** (free):
   - Go to https://visualstudio.microsoft.com/downloads/
   - Download "Visual Studio Community"
   - During installation, select "C++ development tools"

2. **Or install MinGW-w64** (lighter option):
   - Go to https://www.mingw-w64.org/downloads/
   - Install MSYS2
   - Open MSYS2 terminal and run:
   ```bash
   pacman -S mingw-w64-x86_64-gcc
   pacman -S mingw-w64-x86_64-cmake
   ```

#### On macOS

1. **Install Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

2. **Install CMake** (using Homebrew):
   ```bash
   # First install Homebrew if you don't have it
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Then install CMake
   brew install cmake
   ```

#### On Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install build tools
sudo apt install build-essential cmake git

# Install additional libraries
sudo apt install libpthread-dev
```

#### On Linux (CentOS/RHEL/Fedora)

```bash
# For CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install cmake git

# For Fedora
sudo dnf groupinstall "Development Tools"
sudo dnf install cmake git
```

### Step 2: Download Universal Graph Engine

#### Option A: Download Release (Recommended for Beginners)

1. Go to the releases page
2. Download the latest version for your operating system
3. Extract to a folder like `C:\UniversalGraphEngine` or `~/UniversalGraphEngine`

#### Option B: Clone from Git (For Developers)

```bash
# Clone the repository
git clone https://github.com/universal-graph-engine/universal-graph-engine.git

# Enter the directory
cd universal-graph-engine
```

### Step 3: Build the Software

#### Using CMake (All Platforms)

1. **Open Terminal/Command Prompt** in the Universal Graph Engine folder

2. **Create build directory**:
   ```bash
   mkdir build
   cd build
   ```

3. **Configure the build**:
   ```bash
   # Basic build
   cmake ..
   
   # Or with all features enabled
   cmake .. -DUG_BUILD_CLI=ON -DUG_BUILD_EXAMPLES=ON -DUG_BUILD_TESTS=ON
   ```

4. **Build the software**:
   ```bash
   # On Windows with Visual Studio
   cmake --build . --config Release
   
   # On macOS/Linux
   make -j4
   ```

5. **Install (optional)**:
   ```bash
   # On Windows (as Administrator)
   cmake --build . --target install
   
   # On macOS/Linux
   sudo make install
   ```

### Step 4: Verify Installation

1. **Test the CLI**:
   ```bash
   # If installed system-wide
   ug_cli --version
   
   # If built locally
   ./ug_cli --version
   ```

2. **Run a simple test**:
   ```bash
   # Start interactive mode
   ./ug_cli
   
   # Try creating a node
   ug> node test "Hello World"
   ug> stats
   ug> quit
   ```

### Troubleshooting Installation

#### Common Issues

**"cmake not found"**
- Install CMake from https://cmake.org/download/
- Make sure it's in your system PATH

**"Compiler not found"**
- Install development tools as described above
- On Windows, make sure you're using the correct terminal (VS Developer Command Prompt)

**"Permission denied"**
- On Linux/macOS, use `sudo` for installation commands
- On Windows, run Command Prompt as Administrator

**Build fails with missing libraries**
- Install development packages: `sudo apt install build-essential` (Linux)
- Install Xcode command line tools: `xcode-select --install` (macOS)

#### Getting Help

If you're stuck:
1. Check the error message carefully
2. Look in the Troubleshooting section below
3. Search for your error online
4. Ask for help in our community forums

---

## Command Line Interface

### What is a Command Line Interface (CLI)?

A CLI is a way to interact with software by typing commands instead of clicking buttons. It might seem old-fashioned, but it's actually very powerful and efficient once you learn it.

**Think of it like texting with your computer** - you type a message (command), and the computer responds.

### Starting the CLI

#### Interactive Mode (Most Common)
```bash
./ug_cli
```
This starts a conversation with the graph engine where you can type commands one at a time.

#### Script Mode (For Automation)
```bash
./ug_cli -f my_script.ug
```
This runs a file full of commands all at once.

#### Batch Mode (For Data Processing)
```bash
echo "node alice; node bob; edge alice bob" | ./ug_cli --batch
```
This sends commands through a "pipe" from another program.

### Basic Commands

#### Creating Nodes

**Syntax**: `node <name> [data] [type]`

```bash
# Simple node with just a name
ug> node alice
SUCCESS: Created node 'alice' with ID 1

# Node with descriptive data
ug> node bob "Bob Johnson - Engineer"
SUCCESS: Created node 'bob' with ID 2

# Node with a number
ug> node age 25
SUCCESS: Created node 'age' with ID 3

# Node with a decimal number
ug> node temperature 98.6
SUCCESS: Created node 'temperature' with ID 4

# Node with true/false value
ug> node is_active true
SUCCESS: Created node 'is_active' with ID 5
```

**What's Happening:**
- The CLI automatically figures out what type of data you're providing
- Each node gets a unique ID number
- You can refer to nodes by their name instead of remembering numbers

#### Creating Edges (Connections)

**Syntax**: `edge <from> <to> [type] [weight]`

```bash
# Simple connection
ug> edge alice bob
SUCCESS: Created edge alice -> bob (type: CONNECTED, weight: 1.00, ID: 1)

# Named relationship
ug> edge alice bob KNOWS
SUCCESS: Created edge alice -> bob (type: KNOWS, weight: 1.00, ID: 2)

# Relationship with strength
ug> edge alice bob FRIENDS 0.9
SUCCESS: Created edge alice -> bob (type: FRIENDS, weight: 0.90, ID: 3)
```

**Understanding Weights:**
- Weight is a number that represents the strength or importance of a relationship
- 0.0 = very weak connection
- 1.0 = very strong connection
- You can use any number, even negative ones for opposite relationships

#### Getting Information

```bash
# List all nodes
ug> nodes
Nodes in graph (3 total):
Name                 ID         Data
----                 --         ----
alice                1          <data>
bob                  2          <data>
age                  3          <data>

# Show graph statistics
ug> stats
Graph Statistics:
  Nodes: 3
  Relationships: 1
  Named nodes: 3
  Graph density: 0.3333

# Get help
ug> help
Universal Graph Engine CLI
==========================
Available commands:
  node         - Create a new node
  edge         - Create an edge between two nodes
  ...

# Get help for specific command
ug> help node
node - Create a new node
Usage: node <name> [data] [type]
Examples:
node alice "Alice Smith"
node count 42 int
node temp 3.14 float
```

### Advanced Operations

#### Insert: Adding Nodes Between Existing Connections

This is like having two friends, Alice and Bob, and introducing them to your mutual friend Charlie who knows both of them.

**Before Insert:**
```
Alice ←→ Bob (direct connection)
```

**After Insert:**
```
Alice ←→ Charlie ←→ Bob (Charlie is in the middle)
```

**Command:**
```bash
# First create a direct connection
ug> node alice "Alice Smith"
ug> node bob "Bob Johnson"
ug> edge alice bob KNOWS 0.8

# Now insert someone in between
ug> insert alice bob charlie "Charlie Brown - Mutual Friend"
SUCCESS: Inserted node 'charlie' between 'alice' and 'bob'
```

**Real-World Examples:**
- **Social**: Introducing a mutual friend
- **Geography**: Adding a city between two others on a route
- **Business**: Adding a middle manager between CEO and employee
- **Technical**: Adding a server between client and database

#### Triangle: Creating Multi-Way Connections

This creates a node that connects to multiple existing nodes, forming a triangle pattern.

**Before Triangle:**
```
Alice    Bob (not connected)
```

**After Triangle:**
```
Alice ←→ Coordinator ←→ Bob
```

**Command:**
```bash
# Create two separate nodes
ug> node alice "Alice Smith"
ug> node bob "Bob Johnson"

# Create a triangle connector
ug> triangle alice bob coordinator "Project Coordinator"
SUCCESS: Created triangle: 'coordinator' connects to both 'alice' and 'bob'
```

**Real-World Examples:**
- **Work**: Project manager connecting team members
- **Social**: Event organizer connecting attendees
- **Technical**: Router connecting multiple computers
- **Family**: Parent connecting children

#### Hyperedges: Connecting Multiple Nodes Simultaneously

Traditional edges connect two nodes. Hyperedges can connect many nodes at once.

```bash
# Create several people
ug> node alice "Alice"
ug> node bob "Bob"
ug> node charlie "Charlie"
ug> node diana "Diana"

# Create a meeting that includes all of them
ug> hyperedge MEETING alice bob charlie diana
SUCCESS: Created hyperedge connecting 4 nodes
```

### Visualization Commands

#### ASCII Art Visualization

```bash
ug> viz
Generating ASCII visualization (layout: circular)...

Graph Visualization (Circular Layout):
┌────────────────────────────────────────┐
│                                        │
│     A                                  │
│                                        │
│                    C                   │
│                                        │
│                                        │
│                B                       │
│                                        │
└────────────────────────────────────────┘

Legend:
  A = alice
  B = bob
  C = charlie
  - = connection
```

#### Export to Files

```bash
# Export to Graphviz format (can be viewed with many tools)
ug> export dot my_graph.dot
SUCCESS: Graph exported to my_graph.dot
Tip: View with 'dot -Tpng my_graph.dot -o graph.png'

# Export to JSON format
ug> export json my_graph.json
SUCCESS: Graph exported to my_graph.json
```

### Working with Script Files

#### Creating a Script File

Create a text file called `my_graph.ug`:

```bash
# This is a comment - lines starting with # are ignored

# Create some people
node alice "Alice Smith - Software Engineer"
node bob "Bob Johnson - Data Scientist"
node charlie "Charlie Brown - Product Manager"

# Create relationships
edge alice bob COLLEAGUES 0.8
edge bob charlie TEAM_MEMBERS 0.9

# Insert a shared project
insert alice bob project_alpha "Project Alpha - Machine Learning"

# Create a team lead connecting them
triangle bob charlie team_lead "Technical Team Lead"

# Show results
stats
nodes
viz
```

#### Running the Script

```bash
./ug_cli -f my_graph.ug
```

This will execute all commands in the file automatically.

### Property Management

#### Setting Properties on Nodes

```bash
# Set different types of properties
ug> set alice age 28
ug> set alice city "New York"
ug> set alice skills "Python,SQL,Docker"
ug> set bob salary 75000.50
ug> set charlie married true
```

#### Getting Properties

```bash
ug> get alice age
alice.age = 28

ug> get alice city
alice.city = "New York"

ug> get bob salary
bob.salary = 75000.50
```

### Graph Analysis

#### Finding Neighbors

```bash
# Find direct neighbors (1 step away)
ug> neighbors alice
Neighbors of 'alice' (depth 1):
  - bob (via COLLEAGUES, weight: 0.8)
  - project_alpha (via CONNECTED, weight: 1.0)

# Find neighbors up to 2 steps away
ug> neighbors alice 2
Neighbors of 'alice' (depth 2):
  - bob (via COLLEAGUES, weight: 0.8)
  - project_alpha (via CONNECTED, weight: 1.0)
  - charlie (via bob -> TEAM_MEMBERS, weight: 0.72)
```

#### Finding Paths

```bash
# Find path between two nodes
ug> path alice charlie
Finding path from 'alice' to 'charlie':
  alice -> project_alpha -> bob -> charlie
  Total distance: 3 steps, combined weight: 0.648
```

### File Operations

#### Saving Your Work

```bash
# Save current graph to a file
ug> save my_work.ug
SUCCESS: Graph saved to my_work.ug

# Load a previously saved graph
ug> load my_work.ug
SUCCESS: Graph loaded from my_work.ug
```

#### Clearing the Graph

```bash
ug> clear
Are you sure you want to clear the entire graph? (y/N): y
SUCCESS: Graph cleared successfully
```

### Getting Help

#### General Help

```bash
ug> help
```
Shows all available commands with brief descriptions.

#### Specific Command Help

```bash
ug> help node
ug> help insert
ug> help triangle
```
Shows detailed help for a specific command, including examples.

#### Quitting

```bash
ug> quit
ug> exit
ug> q
```
Any of these will exit the CLI.

---

## Programming API

### Understanding APIs

**API** stands for "Application Programming Interface" - it's like a menu at a restaurant. The menu tells you what dishes are available and how to order them, but you don't need to know how the kitchen works.

The Universal Graph Engine API tells you what functions are available and how to use them in your own programs.

### Core Data Structures

#### Graph Structure

```c
// This represents an entire graph
typedef struct ug_graph_t ug_graph_t;

// Think of this as a container that holds all your nodes and edges
ug_graph_t* my_graph = ug_create_graph();
```

#### Node and Edge IDs

```c
// These are like unique ID numbers for nodes and edges
typedef uint64_t ug_node_id_t;
typedef uint64_t ug_relationship_id_t;

// Special value meaning "not found" or "invalid"
#define UG_INVALID_ID 0
```

#### Universal Value System

The Universal Graph Engine can store ANY type of data:

```c
typedef enum {
    UG_TYPE_BOOL,       // true/false
    UG_TYPE_INT,        // whole numbers
    UG_TYPE_FLOAT,      // decimal numbers  
    UG_TYPE_DOUBLE,     // high-precision decimals
    UG_TYPE_STRING,     // text
    UG_TYPE_ARRAY,      // list of things
    UG_TYPE_CUSTOM      // your own data types
} ug_type_t;

typedef struct {
    ug_type_t type;
    union {
        bool bool_val;
        int int_val;
        float float_val;
        double double_val;
        char* string_val;
        void* custom_data;
    } data;
} ug_universal_value_t;
```

### Basic Operations

#### Creating and Destroying Graphs

```c
#include "universal_graph.h"

int main() {
    // Create a new graph
    ug_graph_t* graph = ug_create_graph();
    
    if (graph == NULL) {
        printf("Failed to create graph!\n");
        return 1;
    }
    
    // ... use the graph ...
    
    // Always clean up when done
    ug_destroy_graph(graph);
    return 0;
}
```

**Important**: Always call `ug_destroy_graph()` when you're done, or your program will leak memory!

#### Creating Nodes

```c
// Create a node with string data
ug_node_id_t alice = ug_create_string_node(graph, "Alice Smith");

// Create a node with integer data
ug_node_id_t age = ug_create_int_node(graph, 25);

// Create a node with floating-point data
ug_node_id_t temperature = ug_create_float_node(graph, 98.6);

// Create a node with boolean data
ug_node_id_t is_active = ug_create_bool_node(graph, true);

// Check if creation was successful
if (alice == UG_INVALID_ID) {
    printf("Failed to create node!\n");
}
```

#### Creating Edges

```c
// Create a simple edge
ug_relationship_id_t relationship = ug_create_edge(
    graph,           // the graph
    alice,           // from node
    bob,             // to node
    "KNOWS",         // relationship type
    0.8              // weight (strength)
);

// Check if creation was successful
if (relationship == UG_INVALID_ID) {
    printf("Failed to create edge!\n");
}
```

#### Getting Information

```c
// Get number of nodes
size_t node_count = ug_get_node_count(graph);
printf("Graph has %zu nodes\n", node_count);

// Get number of relationships
size_t rel_count = ug_get_relationship_count(graph);
printf("Graph has %zu relationships\n", rel_count);

// Print graph statistics
ug_print_graph_stats(graph);
```

### Advanced Operations

#### Working with Properties

```c
// Set a property on a node
bool success = ug_set_node_property(graph, alice, "age", "28");
if (!success) {
    printf("Failed to set property!\n");
}

// Get a property from a node
char* age_str = ug_get_node_property(graph, alice, "age");
if (age_str != NULL) {
    printf("Alice's age: %s\n", age_str);
    free(age_str);  // Don't forget to free the returned string!
}
```

#### Creating Hyperedges

```c
// Create a hyperedge connecting multiple nodes
ug_node_id_t participants[] = {alice, bob, charlie, diana};
ug_relationship_id_t meeting = ug_create_hyperedge(
    graph,
    participants,
    4,  // number of participants
    "MEETING"
);
```

#### Temporal Operations

```c
// Create a temporal edge (relationship that changes over time)
ug_relationship_id_t temporal_rel = ug_create_temporal_edge(
    graph,
    alice,
    bob,
    "WORKS_WITH",
    1.0,
    1609459200,  // start time (Unix timestamp)
    1640995200   // end time
);
```

### Working with Different Data Types

#### Storing Custom Data

```c
// Define your own data structure
struct Person {
    char name[100];
    int age;
    float height;
};

// Create a person
struct Person alice_data = {"Alice Smith", 28, 5.6};

// Store it in the graph
ug_universal_value_t value;
value.type = UG_TYPE_CUSTOM;
value.data.custom_data = &alice_data;

ug_node_id_t alice = ug_create_node(graph, UG_TYPE_CUSTOM, &value);
```

#### Working with Arrays

```c
// Create an array of numbers
int numbers[] = {1, 2, 3, 4, 5};

ug_universal_value_t array_value;
array_value.type = UG_TYPE_ARRAY;
array_value.data.array_val = numbers;
array_value.size = 5 * sizeof(int);

ug_node_id_t number_list = ug_create_node(graph, UG_TYPE_ARRAY, &array_value);
```

### Error Handling

```c
// Always check return values
ug_graph_t* graph = ug_create_graph();
if (graph == NULL) {
    fprintf(stderr, "ERROR: Could not create graph\n");
    return 1;
}

ug_node_id_t node = ug_create_string_node(graph, "test");
if (node == UG_INVALID_ID) {
    fprintf(stderr, "ERROR: Could not create node\n");
    ug_destroy_graph(graph);
    return 1;
}

// Function succeeded, continue...
```

### Complete Example Program

Here's a complete program that demonstrates the main features:

```c
#include <stdio.h>
#include <stdlib.h>
#include "universal_graph.h"

int main() {
    printf("Creating a social network graph...\n");
    
    // Create the graph
    ug_graph_t* social_network = ug_create_graph();
    if (!social_network) {
        fprintf(stderr, "Failed to create graph!\n");
        return 1;
    }
    
    // Create people
    ug_node_id_t alice = ug_create_string_node(social_network, "Alice Smith");
    ug_node_id_t bob = ug_create_string_node(social_network, "Bob Johnson");
    ug_node_id_t charlie = ug_create_string_node(social_network, "Charlie Brown");
    
    // Check if nodes were created successfully
    if (alice == UG_INVALID_ID || bob == UG_INVALID_ID || charlie == UG_INVALID_ID) {
        fprintf(stderr, "Failed to create nodes!\n");
        ug_destroy_graph(social_network);
        return 1;
    }
    
    printf("Created %zu people\n", ug_get_node_count(social_network));
    
    // Create relationships
    ug_relationship_id_t alice_knows_bob = ug_create_edge(
        social_network, alice, bob, "KNOWS", 0.8
    );
    
    ug_relationship_id_t bob_friends_charlie = ug_create_edge(
        social_network, bob, charlie, "FRIENDS", 0.9
    );
    
    // Add properties
    ug_set_node_property(social_network, alice, "age", "28");
    ug_set_node_property(social_network, alice, "city", "New York");
    ug_set_node_property(social_network, bob, "age", "32");
    ug_set_node_property(social_network, bob, "city", "San Francisco");
    
    // Show what we created
    printf("\nSocial Network Statistics:\n");
    ug_print_graph_stats(social_network);
    
    // Get and display properties
    char* alice_age = ug_get_node_property(social_network, alice, "age");
    char* alice_city = ug_get_node_property(social_network, alice, "city");
    
    if (alice_age && alice_city) {
        printf("\nAlice is %s years old and lives in %s\n", alice_age, alice_city);
        free(alice_age);
        free(alice_city);
    }
    
    // Create a meeting hyperedge
    ug_node_id_t meeting_participants[] = {alice, bob, charlie};
    ug_relationship_id_t team_meeting = ug_create_hyperedge(
        social_network,
        meeting_participants,
        3,
        "TEAM_MEETING"
    );
    
    if (team_meeting != UG_INVALID_ID) {
        printf("Created team meeting with 3 participants\n");
    }
    
    // Export the graph
    bool export_success = ug_export_graph(social_network, "dot", "social_network.dot");
    if (export_success) {
        printf("Graph exported to social_network.dot\n");
        printf("View with: dot -Tpng social_network.dot -o social_network.png\n");
    }
    
    // Clean up
    ug_destroy_graph(social_network);
    printf("Graph destroyed, memory cleaned up\n");
    
    return 0;
}
```

### Compiling Your Program

#### Using GCC (Linux/macOS/Windows with MinGW)

```bash
# Basic compilation
gcc -o my_program my_program.c -luniversal_graph

# With debugging information
gcc -g -o my_program my_program.c -luniversal_graph

# With all warnings enabled
gcc -Wall -Wextra -g -o my_program my_program.c -luniversal_graph
```

#### Using CMake (Recommended)

Create a `CMakeLists.txt` file:

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyGraphProgram)

# Find Universal Graph Engine
find_package(UniversalGraphEngine REQUIRED)

# Create your executable
add_executable(my_program my_program.c)

# Link against Universal Graph Engine
target_link_libraries(my_program UniversalGraphEngine::universal_graph)
```

Then build:

```bash
mkdir build
cd build
cmake ..
make
```

---

## Advanced Features

### Quantum Graph States

The Universal Graph Engine supports quantum-inspired graph operations where nodes and edges can exist in superposition states.

#### What Are Quantum Graphs?

In quantum physics, particles can exist in multiple states simultaneously until they're observed. Quantum graphs apply this concept to data:

- **Superposition**: A node can represent multiple possibilities at once
- **Entanglement**: Changes to one node instantly affect connected nodes
- **Uncertainty**: Relationships can have probability distributions instead of fixed values

#### Creating Quantum Nodes

```c
// Create a node in quantum superposition
ug_node_id_t quantum_node = ug_create_quantum_node(graph);

// Add multiple possible states
ug_add_quantum_state(graph, quantum_node, "alive", 0.6);    // 60% probability
ug_add_quantum_state(graph, quantum_node, "dead", 0.4);     // 40% probability

// Create entangled nodes
ug_node_id_t node1 = ug_create_quantum_node(graph);
ug_node_id_t node2 = ug_create_quantum_node(graph);
ug_create_quantum_entanglement(graph, node1, node2);
```

#### Quantum Measurements

```c
// Collapse the quantum state by measurement
char* observed_state = ug_measure_quantum_node(graph, quantum_node);
printf("Node collapsed to state: %s\n", observed_state);
free(observed_state);
```

### Temporal Evolution

Graphs can evolve over time, with relationships appearing, changing, and disappearing.

#### Time-Based Operations

```c
// Create a temporal graph
ug_graph_t* temporal_graph = ug_create_temporal_graph();

// Add time-stamped events
time_t now = time(NULL);
ug_node_id_t alice = ug_create_string_node(temporal_graph, "Alice");
ug_node_id_t bob = ug_create_string_node(temporal_graph, "Bob");

// Relationship that starts now and lasts for 1 year
ug_create_temporal_edge(temporal_graph, alice, bob, "KNOWS", 1.0, now, now + 31536000);

// Query graph state at specific time
ug_graph_t* past_state = ug_get_graph_at_time(temporal_graph, now - 86400);  // Yesterday
ug_graph_t* future_state = ug_get_graph_at_time(temporal_graph, now + 86400); // Tomorrow
```

#### Causal Relationships

```c
// Create causal chains where one event leads to another
ug_relationship_id_t cause = ug_create_edge(graph, event1, event2, "CAUSES", 0.8);
ug_set_causal_delay(graph, cause, 3600);  // 1 hour delay

// Query causal effects
ug_node_id_t* effects = ug_get_causal_effects(graph, event1, 7200);  // Effects within 2 hours
```

### Meta-Relationships

Create relationships between relationships themselves.

```c
// Create two relationships
ug_relationship_id_t rel1 = ug_create_edge(graph, alice, bob, "KNOWS", 0.8);
ug_relationship_id_t rel2 = ug_create_edge(graph, charlie, diana, "KNOWS", 0.9);

// Create a relationship between the relationships
ug_relationship_id_t meta_rel = ug_create_meta_relationship(
    graph, rel1, rel2, "SIMILAR_TO", 0.7
);
```

### Genetic Evolution

Graphs can evolve using genetic algorithms to optimize their structure.

```c
// Enable genetic evolution
ug_enable_genetic_evolution(graph);

// Define fitness function
typedef double (*fitness_function_t)(ug_graph_t* graph);

double my_fitness_function(ug_graph_t* g) {
    // Return a score based on graph properties
    size_t node_count = ug_get_node_count(g);
    size_t edge_count = ug_get_relationship_count(g);
    return (double)edge_count / node_count;  // Connectivity ratio
}

// Evolve the graph
ug_evolve_graph(graph, my_fitness_function, 100);  // 100 generations
```

### Streaming Operations

Process continuous streams of graph updates in real-time.

```c
// Create a streaming graph
ug_graph_t* stream_graph = ug_create_streaming_graph();

// Set up streaming callbacks
typedef void (*stream_callback_t)(ug_graph_t* graph, ug_stream_event_t* event);

void my_stream_callback(ug_graph_t* graph, ug_stream_event_t* event) {
    switch (event->type) {
        case UG_STREAM_NODE_ADDED:
            printf("New node added: %lu\n", event->node_id);
            break;
        case UG_STREAM_EDGE_ADDED:
            printf("New edge added: %lu\n", event->relationship_id);
            break;
        case UG_STREAM_NODE_DELETED:
            printf("Node deleted: %lu\n", event->node_id);
            break;
    }
}

ug_set_stream_callback(stream_graph, my_stream_callback);

// Start streaming
ug_start_streaming(stream_graph);
```

### Distributed Graph Operations

Scale across multiple machines for very large graphs.

```c
// Create a distributed graph cluster
ug_cluster_t* cluster = ug_create_cluster();
ug_add_cluster_node(cluster, "192.168.1.100", 8080);
ug_add_cluster_node(cluster, "192.168.1.101", 8080);
ug_add_cluster_node(cluster, "192.168.1.102", 8080);

// Create a distributed graph
ug_graph_t* distributed_graph = ug_create_distributed_graph(cluster);

// Operations are automatically distributed across the cluster
ug_node_id_t node = ug_create_string_node(distributed_graph, "Distributed Node");
```

### Custom Algorithms

Implement your own graph algorithms efficiently.

```c
// Define a custom traversal algorithm
typedef bool (*visitor_function_t)(ug_node_id_t node, void* user_data);

bool my_visitor(ug_node_id_t node, void* user_data) {
    int* count = (int*)user_data;
    (*count)++;
    
    // Return true to continue traversal, false to stop
    return true;
}

// Traverse the graph with your custom algorithm
int node_count = 0;
ug_traverse_graph(graph, start_node, my_visitor, &node_count);
printf("Visited %d nodes\n", node_count);
```

### Performance Optimization

#### Memory Pool Allocation

```c
// Create a memory pool for better performance
ug_memory_pool_t* pool = ug_create_memory_pool(1024 * 1024);  // 1MB pool
ug_graph_t* graph = ug_create_graph_with_pool(pool);

// Graph operations will use the pool for faster allocation
// ... use graph ...

// Clean up
ug_destroy_graph(graph);
ug_destroy_memory_pool(pool);
```

#### Parallel Processing

```c
// Enable parallel processing (requires OpenMP)
ug_set_thread_count(graph, 4);  // Use 4 threads

// Some operations will automatically use multiple threads
ug_parallel_traverse(graph, start_node, visitor_func, user_data);
```

#### Caching

```c
// Enable result caching for expensive operations
ug_enable_query_cache(graph, 1000);  // Cache up to 1000 results

// Repeated queries will be served from cache
ug_node_id_t* path1 = ug_find_shortest_path(graph, alice, bob);
ug_node_id_t* path2 = ug_find_shortest_path(graph, alice, bob);  // Served from cache
```

---

## Examples and Tutorials

### Tutorial 1: Your First Graph Program

Let's build a simple family tree to understand the basics.

#### Step 1: Set Up Your Development Environment

Create a new file called `family_tree.c`:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "universal_graph.h"

int main() {
    printf("Building a Family Tree\n");
    printf("======================\n\n");
    
    // We'll add code here step by step
    
    return 0;
}
```

#### Step 2: Create the Graph and Add People

```c
int main() {
    printf("Building a Family Tree\n");
    printf("======================\n\n");
    
    // Create a new graph to represent our family
    ug_graph_t* family = ug_create_graph();
    if (!family) {
        printf("ERROR: Couldn't create family tree!\n");
        return 1;
    }
    
    // Add family members
    printf("Adding family members...\n");
    ug_node_id_t grandpa = ug_create_string_node(family, "Grandpa John");
    ug_node_id_t grandma = ug_create_string_node(family, "Grandma Mary");
    ug_node_id_t dad = ug_create_string_node(family, "Dad Mike");
    ug_node_id_t mom = ug_create_string_node(family, "Mom Sarah");
    ug_node_id_t alice = ug_create_string_node(family, "Alice");
    ug_node_id_t bob = ug_create_string_node(family, "Bob");
    
    // Check if everyone was added successfully
    if (grandpa == UG_INVALID_ID || grandma == UG_INVALID_ID || 
        dad == UG_INVALID_ID || mom == UG_INVALID_ID || 
        alice == UG_INVALID_ID || bob == UG_INVALID_ID) {
        printf("ERROR: Couldn't add all family members!\n");
        ug_destroy_graph(family);
        return 1;
    }
    
    printf("Added %zu family members\n\n", ug_get_node_count(family));
    
    // Clean up and exit
    ug_destroy_graph(family);
    return 0;
}
```

#### Step 3: Add Relationships

```c
    // Add relationships between family members
    printf("Adding family relationships...\n");
    
    // Marriages
    ug_create_edge(family, grandpa, grandma, "MARRIED_TO", 1.0);
    ug_create_edge(family, dad, mom, "MARRIED_TO", 1.0);
    
    // Parent-child relationships
    ug_create_edge(family, grandpa, dad, "PARENT_OF", 1.0);
    ug_create_edge(family, grandma, dad, "PARENT_OF", 1.0);
    ug_create_edge(family, dad, alice, "PARENT_OF", 1.0);
    ug_create_edge(family, dad, bob, "PARENT_OF", 1.0);
    ug_create_edge(family, mom, alice, "PARENT_OF", 1.0);
    ug_create_edge(family, mom, bob, "PARENT_OF", 1.0);
    
    // Sibling relationship
    ug_create_edge(family, alice, bob, "SIBLING_OF", 1.0);
    
    printf("Added %zu relationships\n\n", ug_get_relationship_count(family));
```

#### Step 4: Add Personal Information

```c
    // Add ages and other information
    printf("Adding personal information...\n");
    ug_set_node_property(family, grandpa, "age", "75");
    ug_set_node_property(family, grandpa, "job", "Retired Teacher");
    
    ug_set_node_property(family, grandma, "age", "72");
    ug_set_node_property(family, grandma, "job", "Retired Nurse");
    
    ug_set_node_property(family, dad, "age", "45");
    ug_set_node_property(family, dad, "job", "Software Engineer");
    
    ug_set_node_property(family, mom, "age", "42");
    ug_set_node_property(family, mom, "job", "Doctor");
    
    ug_set_node_property(family, alice, "age", "16");
    ug_set_node_property(family, alice, "school", "High School");
    
    ug_set_node_property(family, bob, "age", "14");
    ug_set_node_property(family, bob, "school", "Middle School");
    
    printf("Added personal information\n\n");
```

#### Step 5: Display Information

```c
    // Show family tree statistics
    printf("Family Tree Statistics:\n");
    ug_print_graph_stats(family);
    printf("\n");
    
    // Show some personal information
    char* dad_age = ug_get_node_property(family, dad, "age");
    char* dad_job = ug_get_node_property(family, dad, "job");
    char* alice_age = ug_get_node_property(family, alice, "age");
    
    if (dad_age && dad_job && alice_age) {
        printf("Dad Mike is %s years old and works as a %s\n", dad_age, dad_job);
        printf("Alice is %s years old\n", alice_age);
        
        // Don't forget to free the strings!
        free(dad_age);
        free(dad_job);
        free(alice_age);
    }
    
    // Export the family tree
    printf("\nExporting family tree...\n");
    if (ug_export_graph(family, "dot", "family_tree.dot")) {
        printf("Family tree exported to family_tree.dot\n");
        printf("View with: dot -Tpng family_tree.dot -o family_tree.png\n");
    }
```

#### Complete Program

Here's the complete `family_tree.c`:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "universal_graph.h"

int main() {
    printf("Building a Family Tree\n");
    printf("======================\n\n");
    
    // Create a new graph to represent our family
    ug_graph_t* family = ug_create_graph();
    if (!family) {
        printf("ERROR: Couldn't create family tree!\n");
        return 1;
    }
    
    // Add family members
    printf("Adding family members...\n");
    ug_node_id_t grandpa = ug_create_string_node(family, "Grandpa John");
    ug_node_id_t grandma = ug_create_string_node(family, "Grandma Mary");
    ug_node_id_t dad = ug_create_string_node(family, "Dad Mike");
    ug_node_id_t mom = ug_create_string_node(family, "Mom Sarah");
    ug_node_id_t alice = ug_create_string_node(family, "Alice");
    ug_node_id_t bob = ug_create_string_node(family, "Bob");
    
    // Check if everyone was added successfully
    if (grandpa == UG_INVALID_ID || grandma == UG_INVALID_ID || 
        dad == UG_INVALID_ID || mom == UG_INVALID_ID || 
        alice == UG_INVALID_ID || bob == UG_INVALID_ID) {
        printf("ERROR: Couldn't add all family members!\n");
        ug_destroy_graph(family);
        return 1;
    }
    
    printf("Added %zu family members\n\n", ug_get_node_count(family));
    
    // Add relationships between family members
    printf("Adding family relationships...\n");
    
    // Marriages
    ug_create_edge(family, grandpa, grandma, "MARRIED_TO", 1.0);
    ug_create_edge(family, dad, mom, "MARRIED_TO", 1.0);
    
    // Parent-child relationships
    ug_create_edge(family, grandpa, dad, "PARENT_OF", 1.0);
    ug_create_edge(family, grandma, dad, "PARENT_OF", 1.0);
    ug_create_edge(family, dad, alice, "PARENT_OF", 1.0);
    ug_create_edge(family, dad, bob, "PARENT_OF", 1.0);
    ug_create_edge(family, mom, alice, "PARENT_OF", 1.0);
    ug_create_edge(family, mom, bob, "PARENT_OF", 1.0);
    
    // Sibling relationship
    ug_create_edge(family, alice, bob, "SIBLING_OF", 1.0);
    
    printf("Added %zu relationships\n\n", ug_get_relationship_count(family));
    
    // Add ages and other information
    printf("Adding personal information...\n");
    ug_set_node_property(family, grandpa, "age", "75");
    ug_set_node_property(family, grandpa, "job", "Retired Teacher");
    
    ug_set_node_property(family, grandma, "age", "72");
    ug_set_node_property(family, grandma, "job", "Retired Nurse");
    
    ug_set_node_property(family, dad, "age", "45");
    ug_set_node_property(family, dad, "job", "Software Engineer");
    
    ug_set_node_property(family, mom, "age", "42");
    ug_set_node_property(family, mom, "job", "Doctor");
    
    ug_set_node_property(family, alice, "age", "16");
    ug_set_node_property(family, alice, "school", "High School");
    
    ug_set_node_property(family, bob, "age", "14");
    ug_set_node_property(family, bob, "school", "Middle School");
    
    printf("Added personal information\n\n");
    
    // Show family tree statistics
    printf("Family Tree Statistics:\n");
    ug_print_graph_stats(family);
    printf("\n");
    
    // Show some personal information
    char* dad_age = ug_get_node_property(family, dad, "age");
    char* dad_job = ug_get_node_property(family, dad, "job");
    char* alice_age = ug_get_node_property(family, alice, "age");
    
    if (dad_age && dad_job && alice_age) {
        printf("Dad Mike is %s years old and works as a %s\n", dad_age, dad_job);
        printf("Alice is %s years old\n", alice_age);
        
        // Don't forget to free the strings!
        free(dad_age);
        free(dad_job);
        free(alice_age);
    }
    
    // Export the family tree
    printf("\nExporting family tree...\n");
    if (ug_export_graph(family, "dot", "family_tree.dot")) {
        printf("Family tree exported to family_tree.dot\n");
        printf("View with: dot -Tpng family_tree.dot -o family_tree.png\n");
    }
    
    // Clean up
    ug_destroy_graph(family);
    printf("\nFamily tree complete!\n");
    
    return 0;
}
```

#### Compiling and Running

```bash
# Compile the program
gcc -o family_tree family_tree.c -luniversal_graph

# Run it
./family_tree

# View the visual output (if you have Graphviz installed)
dot -Tpng family_tree.dot -o family_tree.png
```

### Tutorial 2: Social Network Analysis

This tutorial builds a more complex social network and analyzes it.

#### The Social Network

We'll model a small company with employees, departments, and projects.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "universal_graph.h"

// Helper function to create a person with properties
ug_node_id_t create_person(ug_graph_t* graph, const char* name, const char* department, const char* role) {
    ug_node_id_t person = ug_create_string_node(graph, name);
    if (person != UG_INVALID_ID) {
        ug_set_node_property(graph, person, "department", department);
        ug_set_node_property(graph, person, "role", role);
    }
    return person;
}

int main() {
    printf("Company Social Network Analysis\n");
    printf("===============================\n\n");
    
    // Create the graph
    ug_graph_t* company = ug_create_graph();
    if (!company) {
        printf("ERROR: Could not create company graph\n");
        return 1;
    }
    
    // Create employees
    printf("Adding employees...\n");
    ug_node_id_t alice = create_person(company, "Alice Johnson", "Engineering", "Senior Developer");
    ug_node_id_t bob = create_person(company, "Bob Smith", "Engineering", "Junior Developer");
    ug_node_id_t charlie = create_person(company, "Charlie Brown", "Product", "Product Manager");
    ug_node_id_t diana = create_person(company, "Diana Prince", "Design", "UX Designer");
    ug_node_id_t eve = create_person(company, "Eve Wilson", "Marketing", "Marketing Manager");
    ug_node_id_t frank = create_person(company, "Frank Miller", "Executive", "CEO");
    
    printf("Added %zu employees\n\n", ug_get_node_count(company));
    
    // Create work relationships
    printf("Adding work relationships...\n");
    ug_create_edge(company, alice, bob, "MENTORS", 0.9);
    ug_create_edge(company, charlie, alice, "COLLABORATES_WITH", 0.8);
    ug_create_edge(company, charlie, diana, "WORKS_WITH", 0.7);
    ug_create_edge(company, diana, eve, "COORDINATES_WITH", 0.6);
    ug_create_edge(company, frank, charlie, "MANAGES", 1.0);
    ug_create_edge(company, frank, alice, "REPORTS_FROM", 0.5);
    
    // Create cross-departmental relationships
    ug_create_edge(company, alice, diana, "TECHNICAL_REVIEWS", 0.4);
    ug_create_edge(company, bob, eve, "SUPPORTS", 0.3);
    
    printf("Added %zu work relationships\n\n", ug_get_relationship_count(company));
    
    // Create projects as hyperedges
    printf("Adding projects...\n");
    ug_node_id_t project_team1[] = {alice, bob, charlie};
    ug_relationship_id_t project_alpha = ug_create_hyperedge(
        company, project_team1, 3, "PROJECT_ALPHA"
    );
    
    ug_node_id_t project_team2[] = {charlie, diana, eve};
    ug_relationship_id_t project_beta = ug_create_hyperedge(
        company, project_team2, 3, "PROJECT_BETA"
    );
    
    printf("Added 2 projects\n\n");
    
    // Analyze the network
    printf("Network Analysis:\n");
    printf("-----------------\n");
    ug_print_graph_stats(company);
    
    // Find the most connected person (simple version)
    size_t node_count = ug_get_node_count(company);
    printf("\nEmployee Connectivity Analysis:\n");
    
    // This is a simplified analysis - in a real implementation,
    // you'd use the graph traversal functions
    printf("- Alice: Central figure in engineering\n");
    printf("- Charlie: Bridge between technical and business\n");
    printf("- Frank: Executive oversight\n\n");
    
    // Export for visualization
    printf("Exporting network visualization...\n");
    if (ug_export_graph(company, "dot", "company_network.dot")) {
        printf("Network exported to company_network.dot\n");
        printf("Generate image with: dot -Tpng company_network.dot -o network.png\n");
    }
    
    // Clean up
    ug_destroy_graph(company);
    printf("\nAnalysis complete!\n");
    
    return 0;
}
```

### Tutorial 3: Recommendation System

This tutorial shows how to build a simple recommendation system using graph relationships.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "universal_graph.h"

// Structure to hold recommendation results
typedef struct {
    ug_node_id_t item_id;
    double score;
    char* item_name;
} recommendation_t;

// Simple recommendation based on common connections
void find_recommendations(ug_graph_t* graph, ug_node_id_t user, 
                         recommendation_t* recommendations, int max_recommendations) {
    // This is a simplified recommendation algorithm
    // In practice, you'd implement collaborative filtering or content-based filtering
    
    printf("Finding recommendations for user %lu...\n", user);
    
    // For this example, we'll create some dummy recommendations
    recommendations[0].item_id = 101;
    recommendations[0].score = 0.95;
    recommendations[0].item_name = strdup("The Matrix");
    
    recommendations[1].item_id = 102;
    recommendations[1].score = 0.87;
    recommendations[1].item_name = strdup("Inception");
    
    recommendations[2].item_id = 103;
    recommendations[2].score = 0.82;
    recommendations[2].item_name = strdup("Interstellar");
}

int main() {
    printf("Movie Recommendation System\n");
    printf("===========================\n\n");
    
    // Create the graph
    ug_graph_t* movie_db = ug_create_graph();
    if (!movie_db) {
        printf("ERROR: Could not create movie database\n");
        return 1;
    }
    
    // Create users
    printf("Adding users...\n");
    ug_node_id_t alice = ug_create_string_node(movie_db, "Alice");
    ug_node_id_t bob = ug_create_string_node(movie_db, "Bob");
    ug_node_id_t charlie = ug_create_string_node(movie_db, "Charlie");
    
    // Create movies
    ug_node_id_t matrix = ug_create_string_node(movie_db, "The Matrix");
    ug_node_id_t inception = ug_create_string_node(movie_db, "Inception");
    ug_node_id_t interstellar = ug_create_string_node(movie_db, "Interstellar");
    ug_node_id_t avatar = ug_create_string_node(movie_db, "Avatar");
    ug_node_id_t titanic = ug_create_string_node(movie_db, "Titanic");
    
    printf("Added %zu users and movies\n\n", ug_get_node_count(movie_db));
    
    // Create rating relationships (user RATED movie with score)
    printf("Adding movie ratings...\n");
    ug_create_edge(movie_db, alice, matrix, "RATED", 5.0);      // Alice loves The Matrix
    ug_create_edge(movie_db, alice, inception, "RATED", 4.5);   // Alice likes Inception
    ug_create_edge(movie_db, alice, avatar, "RATED", 3.0);      // Alice is neutral on Avatar
    
    ug_create_edge(movie_db, bob, matrix, "RATED", 4.8);        // Bob loves The Matrix
    ug_create_edge(movie_db, bob, interstellar, "RATED", 4.9);  // Bob loves Interstellar
    ug_create_edge(movie_db, bob, titanic, "RATED", 2.0);       // Bob dislikes Titanic
    
    ug_create_edge(movie_db, charlie, inception, "RATED", 4.7); // Charlie loves Inception
    ug_create_edge(movie_db, charlie, interstellar, "RATED", 4.6); // Charlie loves Interstellar
    ug_create_edge(movie_db, charlie, avatar, "RATED", 4.2);    // Charlie likes Avatar
    
    printf("Added %zu ratings\n\n", ug_get_relationship_count(movie_db));
    
    // Find recommendations for Alice
    printf("Generating recommendations for Alice...\n");
    recommendation_t recommendations[3];
    find_recommendations(movie_db, alice, recommendations, 3);
    
    printf("\nRecommendations for Alice:\n");
    for (int i = 0; i < 3; i++) {
        printf("%d. %s (Score: %.2f)\n", i+1, recommendations[i].item_name, recommendations[i].score);
        free(recommendations[i].item_name);
    }
    
    // Export the recommendation graph
    printf("\nExporting recommendation graph...\n");
    if (ug_export_graph(movie_db, "dot", "movie_recommendations.dot")) {
        printf("Graph exported to movie_recommendations.dot\n");
    }
    
    // Clean up
    ug_destroy_graph(movie_db);
    printf("\nRecommendation system complete!\n");
    
    return 0;
}
```

### Tutorial 4: Using the CLI for Complex Scenarios

This tutorial shows how to use the CLI to model complex real-world scenarios.

#### Scenario: University Course Prerequisites

Create a file called `university.ug`:

```bash
# University Course Prerequisite System
# ====================================

# Create foundational courses
node math101 "Mathematics 101 - Basic Algebra"
node math102 "Mathematics 102 - Calculus I"
node math201 "Mathematics 201 - Calculus II"
node math301 "Mathematics 301 - Linear Algebra"

node cs101 "Computer Science 101 - Programming Basics"
node cs102 "Computer Science 102 - Data Structures"
node cs201 "Computer Science 201 - Algorithms"
node cs301 "Computer Science 301 - Database Systems"

node phys101 "Physics 101 - Mechanics"
node phys102 "Physics 102 - Electricity & Magnetism"

# Create prerequisite relationships
edge math101 math102 PREREQUISITE 1.0
edge math102 math201 PREREQUISITE 1.0
edge math201 math301 PREREQUISITE 1.0

edge cs101 cs102 PREREQUISITE 1.0
edge cs102 cs201 PREREQUISITE 1.0
edge cs201 cs301 PREREQUISITE 1.0

edge math101 phys101 PREREQUISITE 1.0
edge phys101 phys102 PREREQUISITE 1.0

# Cross-disciplinary prerequisites
edge math102 cs201 PREREQUISITE 0.8
edge math301 cs301 PREREQUISITE 0.6

# Insert bridging courses where students need extra preparation
insert math102 cs201 discrete_math "Discrete Mathematics"
insert phys101 cs301 numerical_methods "Numerical Methods"

# Create study groups that connect multiple courses
triangle math201 cs201 study_group_advanced "Advanced Mathematics & CS Study Group"
triangle phys102 cs301 research_group "Physics-CS Research Group"

# Add course properties
set math101 credits 3
set math101 difficulty "Beginner"
set math102 credits 4
set math102 difficulty "Intermediate"

set cs101 credits 3
set cs101 difficulty "Beginner"
set cs201 credits 4
set cs201 difficulty "Advanced"

# Show the course network
stats
nodes

# Visualize the prerequisite network
viz

# Export for academic planning
export dot university_prerequisites.dot
```

Run this with:
```bash
./ug_cli -f university.ug
```

#### Scenario: Supply Chain Management

Create `supply_chain.ug`:

```bash
# Global Supply Chain Network
# ===========================

# Create suppliers
node supplier_a "Supplier A - Raw Materials (China)"
node supplier_b "Supplier B - Components (Germany)"
node supplier_c "Supplier C - Electronics (Japan)"

# Create manufacturers
node factory_1 "Factory 1 - Assembly (Mexico)"
node factory_2 "Factory 2 - Quality Control (USA)"

# Create distributors
node distributor_east "East Coast Distributor"
node distributor_west "West Coast Distributor"
node distributor_europe "European Distributor"

# Create retailers
node retailer_1 "Retailer 1 - Big Box Store"
node retailer_2 "Retailer 2 - Online Platform"
node retailer_3 "Retailer 3 - Specialty Store"

# Create supply relationships with capacity weights
edge supplier_a factory_1 SUPPLIES 0.8
edge supplier_b factory_1 SUPPLIES 0.6
edge supplier_c factory_2 SUPPLIES 0.9

edge factory_1 factory_2 TRANSFERS 0.7
edge factory_2 distributor_east SHIPS 0.9
edge factory_2 distributor_west SHIPS 0.8
edge factory_2 distributor_europe SHIPS 0.6

# Insert logistics hubs where needed
insert factory_2 distributor_europe logistics_hub "International Logistics Hub"
insert distributor_east retailer_1 warehouse_1 "Regional Warehouse 1"
insert distributor_west retailer_2 warehouse_2 "Regional Warehouse 2"

# Create retail triangles (distributors serving multiple retailers)
triangle distributor_east distributor_west fulfillment_center "Cross-Country Fulfillment"
triangle retailer_1 retailer_2 customer_service "Shared Customer Service"

# Add logistics properties
set supplier_a lead_time "14 days"
set supplier_a capacity "10000 units/month"
set factory_1 production_rate "5000 units/week"
set logistics_hub customs_time "3 days"

# Analyze the supply chain
stats
nodes

# Find potential bottlenecks (nodes with many connections)
neighbors factory_2
neighbors logistics_hub

# Export for supply chain analysis
export json supply_chain.json
export dot supply_chain.dot
```

These examples show how the Universal Graph Engine can model complex real-world systems with interconnected entities and relationships.

---

## Language Migration Guide

### Migrating from C to C++

The Universal Graph Engine provides zero-cost C++ wrappers that make the code more modern and safe.

#### C++ Wrapper Features

```cpp
#include "universal_graph.hpp"
using namespace ug;

int main() {
    // RAII - automatic cleanup, no manual destroy calls needed
    UniversalGraph graph;
    
    // Type-safe node creation
    auto alice = graph.create_node<std::string>("Alice Smith");
    auto age = graph.create_node<int>(28);
    auto temperature = graph.create_node<double>(98.6);
    
    // Fluent API for building graphs
    auto relationship = graph.create_edge(alice, age, "HAS_AGE", 1.0);
    
    // Exception handling instead of checking return codes
    try {
        auto path = graph.find_shortest_path(alice, age);
        std::cout << "Path found with " << path.size() << " steps\n";
    } catch (const GraphException& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    
    // Modern C++ features
    graph.for_each_node([](const Node& node) {
        std::cout << "Node: " << node.get_data<std::string>() << "\n";
    });
    
    return 0;
}
```

#### Builder Pattern in C++

```cpp
// Complex graph construction with builder pattern
auto social_network = GraphBuilder()
    .add_node("alice", "Alice Smith")
    .add_node("bob", "Bob Johnson")
    .add_node("charlie", "Charlie Brown")
    .add_edge("alice", "bob", "KNOWS", 0.8)
    .add_edge("bob", "charlie", "FRIENDS", 0.9)
    .insert_between("alice", "bob", "mutual_friend", "Dave Wilson")
    .create_triangle("alice", "charlie", "coordinator", "Project Lead")
    .build();
```

#### Template-Based Type Safety

```cpp
// Generic graph algorithms
template<typename T>
class TypedGraph : public UniversalGraph {
public:
    NodeId create_typed_node(const T& data) {
        return create_node<T>(data);
    }
    
    std::optional<T> get_node_data(NodeId id) {
        try {
            return get_node<T>(id).get_data();
        } catch (...) {
            return std::nullopt;
        }
    }
};

// Usage
TypedGraph<Person> people_graph;
auto alice_id = people_graph.create_typed_node(Person{"Alice", 28, "Engineer"});
```

### Migrating to Rust

Rust provides memory safety guarantees and zero-cost abstractions.

#### Rust Bindings Features

```rust
use universal_graph::prelude::*;

fn main() -> UgResult<()> {
    // Safe graph creation with automatic cleanup
    let graph = UniversalGraph::new()?;
    
    // Type-safe node creation
    let alice = graph.create_node("Alice Smith".into())?;
    let bob = graph.create_node("Bob Johnson".into())?;
    
    // Result-based error handling
    let relationship = graph.create_edge(alice, bob, "KNOWS", 1.0)?;
    
    // Pattern matching for robust error handling
    match graph.find_shortest_path(alice, bob) {
        Ok(path) => println!("Found path with {} steps", path.len()),
        Err(UgError::NotFound) => println!("No path exists"),
        Err(e) => eprintln!("Error: {}", e),
    }
    
    // Iterator-based operations
    graph.nodes()
        .filter(|node| node.has_property("age"))
        .for_each(|node| println!("Node: {:?}", node));
    
    Ok(())
}
```

#### Builder Pattern in Rust

```rust
// Fluent API with compile-time safety
let social_network = GraphBuilder::new()?
    .add_node("alice", "Alice Smith")?
    .add_node("bob", "Bob Johnson")?
    .add_edge("alice", "bob", "KNOWS", 0.8)?
    .insert_between("alice", "bob", "bridge", "Bridge Node")?
    .create_triangle("alice", "bob", "connector", "Connector Node")?
    .build();
```

#### Safe Memory Management

```rust
// Automatic memory management - no manual cleanup needed
fn process_large_graph() -> UgResult<()> {
    let graph = UniversalGraph::new()?;
    
    // Add millions of nodes - memory is automatically managed
    for i in 0..1_000_000 {
        graph.create_node(format!("Node {}", i))?;
    }
    
    // Graph is automatically cleaned up when it goes out of scope
    Ok(())
} // <- Memory automatically freed here
```

### Migrating to Go

Go provides simple concurrency and garbage collection.

```go
package main

import (
    "fmt"
    "log"
    "github.com/universal-graph-engine/go-bindings/graph"
)

func main() {
    // Simple graph creation with garbage collection
    g, err := graph.New()
    if err != nil {
        log.Fatal("Failed to create graph:", err)
    }
    defer g.Close() // Explicit cleanup for C resources
    
    // Simple error handling
    alice, err := g.CreateNode("Alice Smith")
    if err != nil {
        log.Fatal("Failed to create node:", err)
    }
    
    bob, err := g.CreateNode("Bob Johnson")
    if err != nil {
        log.Fatal("Failed to create node:", err)
    }
    
    // Create relationship
    _, err = g.CreateEdge(alice, bob, "KNOWS", 0.8)
    if err != nil {
        log.Fatal("Failed to create edge:", err)
    }
    
    // Concurrent operations
    go func() {
        neighbors, err := g.GetNeighbors(alice)
        if err != nil {
            log.Printf("Error getting neighbors: %v", err)
            return
        }
        fmt.Printf("Alice has %d neighbors\n", len(neighbors))
    }()
    
    // Channel-based streaming
    updates := make(chan graph.Update, 100)
    g.EnableStreaming(updates)
    
    go func() {
        for update := range updates {
            fmt.Printf("Graph update: %+v\n", update)
        }
    }()
    
    // Export the graph
    err = g.Export("dot", "social_network.dot")
    if err != nil {
        log.Printf("Export failed: %v", err)
    }
}
```

### Migrating to Python

Python provides the easiest interface for rapid prototyping and data science.

```python
from universal_graph import UniversalGraph, GraphBuilder
import pandas as pd
import networkx as nx

def main():
    # Simple graph creation
    graph = UniversalGraph()
    
    # Pythonic node creation
    alice = graph.create_node("Alice Smith")
    bob = graph.create_node("Bob Johnson")
    charlie = graph.create_node("Charlie Brown")
    
    # Easy relationship creation
    graph.create_edge(alice, bob, "KNOWS", weight=0.8)
    graph.create_edge(bob, charlie, "FRIENDS", weight=0.9)
    
    # Dictionary-like property access
    graph.nodes[alice]["age"] = 28
    graph.nodes[alice]["city"] = "New York"
    
    # List comprehensions for analysis
    young_people = [node for node in graph.nodes 
                   if graph.nodes[node].get("age", 0) < 30]
    
    # Pandas integration for data analysis
    df = graph.to_dataframe()
    print(df.describe())
    
    # NetworkX compatibility
    nx_graph = graph.to_networkx()
    centrality = nx.betweenness_centrality(nx_graph)
    print("Most central person:", max(centrality, key=centrality.get))
    
    # Easy visualization
    graph.visualize(layout="spring", save_as="social_network.png")
    
    # Machine learning integration
    from sklearn.cluster import SpectralClustering
    
    adjacency_matrix = graph.to_adjacency_matrix()
    clustering = SpectralClustering(n_clusters=2, random_state=0)
    clusters = clustering.fit_predict(adjacency_matrix)
    
    print("Detected communities:", clusters)

if __name__ == "__main__":
    main()
```

### Migration Strategy

#### Phase 1: Assessment (Week 1)
1. **Inventory Current Code**: List all C functions you're using
2. **Identify Dependencies**: What other libraries do you use?
3. **Performance Requirements**: Do you need the same performance?
4. **Team Skills**: What languages does your team know?

#### Phase 2: Gradual Migration (Weeks 2-4)
1. **Start with New Features**: Write new code in the target language
2. **Create Wrapper Layer**: Keep existing C code, wrap it with new language
3. **Test Thoroughly**: Ensure behavior remains identical
4. **Document Differences**: Note any API changes

#### Phase 3: Full Migration (Weeks 5-8)
1. **Convert Core Logic**: Migrate main algorithms
2. **Update Data Structures**: Use target language's native types
3. **Optimize for Target Language**: Use language-specific best practices
4. **Performance Testing**: Ensure acceptable performance

#### Phase 4: Optimization (Weeks 9-12)
1. **Language-Specific Optimizations**: Use advanced features
2. **Integration with Ecosystem**: Use language-specific libraries
3. **Documentation Update**: Update all documentation
4. **Team Training**: Train team on new codebase

### Common Migration Challenges

#### Memory Management
- **C to C++**: Use RAII and smart pointers
- **C to Rust**: Use ownership system and borrowing
- **C to Go**: Use garbage collector, be careful with C interop
- **C to Python**: Let Python handle memory, use context managers

#### Error Handling
- **C**: Return codes and NULL checks
- **C++**: Exceptions and optional types
- **Rust**: Result<T, E> and Option<T>
- **Go**: Multiple return values with error
- **Python**: Exceptions with try/except

#### Performance Considerations
- **C++**: Usually similar performance to C
- **Rust**: Zero-cost abstractions, similar to C performance
- **Go**: Good performance, some GC overhead
- **Python**: Slower, but excellent for prototyping

#### Best Practices for Each Language

**C++ Migration:**
```cpp
// Use RAII for resource management
class GraphManager {
    std::unique_ptr<ug_graph_t, GraphDeleter> graph_;
public:
    GraphManager() : graph_(ug_create_graph(), GraphDeleter{}) {}
    // Automatic cleanup in destructor
};

// Use exceptions for error handling
void safe_create_node(UniversalGraph& graph, const std::string& data) {
    auto node = graph.create_node(data);
    if (!node) {
        throw std::runtime_error("Failed to create node");
    }
}
```

**Rust Migration:**
```rust
// Use Result for error handling
fn create_social_network() -> Result<UniversalGraph, UgError> {
    let graph = UniversalGraph::new()?;
    let alice = graph.create_node("Alice".into())?;
    let bob = graph.create_node("Bob".into())?;
    graph.create_edge(alice, bob, "KNOWS", 1.0)?;
    Ok(graph)
}

// Use ownership to prevent data races
fn analyze_graph(graph: &UniversalGraph) -> Vec<NodeId> {
    graph.nodes().filter(|n| n.degree() > 3).collect()
}
```

---

## Troubleshooting

### Common Installation Issues

#### Issue: "cmake not found"
**Symptoms**: 
```
bash: cmake: command not found
```

**Solutions**:
1. **Install CMake**:
   - Windows: Download from https://cmake.org/download/
   - macOS: `brew install cmake`
   - Linux: `sudo apt install cmake` or `sudo yum install cmake`

2. **Add to PATH**:
   - Windows: Add CMake bin directory to system PATH
   - macOS/Linux: CMake is usually added automatically

3. **Verify Installation**:
   ```bash
   cmake --version
   ```

#### Issue: "Compiler not found"
**Symptoms**:
```
CMake Error: CMAKE_C_COMPILER not set
```

**Solutions**:
1. **Install Development Tools**:
   - Windows: Install Visual Studio Community or MinGW
   - macOS: `xcode-select --install`
   - Linux: `sudo apt install build-essential`

2. **Set Compiler Manually**:
   ```bash
   cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
   ```

#### Issue: "Permission denied"
**Symptoms**:
```
mkdir: cannot create directory 'build': Permission denied
```

**Solutions**:
1. **Check Directory Permissions**:
   ```bash
   ls -la
   chmod 755 .
   ```

2. **Run as Administrator/Root** (if necessary):
   - Windows: Right-click Command Prompt → "Run as administrator"
   - macOS/Linux: `sudo cmake --build . --target install`

### Common Compilation Issues

#### Issue: "undefined reference to ug_create_graph"
**Symptoms**:
```
/usr/bin/ld: undefined reference to `ug_create_graph'
```

**Solutions**:
1. **Link Against Library**:
   ```bash
   gcc -o my_program my_program.c -luniversal_graph
   ```

2. **Check Library Installation**:
   ```bash
   # Find the library
   find /usr -name "*universal_graph*" 2>/dev/null
   
   # Check if it's in the linker path
   ldconfig -p | grep universal_graph
   ```

3. **Set Library Path**:
   ```bash
   export LD_LIBRARY_PATH=/path/to/universal_graph/lib:$LD_LIBRARY_PATH
   ```

#### Issue: "fatal error: universal_graph.h: No such file or directory"
**Symptoms**:
```
fatal error: universal_graph.h: No such file or directory
```

**Solutions**:
1. **Check Include Path**:
   ```bash
   gcc -I/path/to/headers -o my_program my_program.c -luniversal_graph
   ```

2. **Install Headers**:
   ```bash
   sudo make install  # This should install headers
   ```

3. **Use pkg-config** (if available):
   ```bash
   gcc $(pkg-config --cflags --libs universal-graph) -o my_program my_program.c
   ```

### Runtime Issues

#### Issue: "Segmentation fault"
**Symptoms**:
Program crashes with segmentation fault.

**Debugging Steps**:
1. **Use Debugger**:
   ```bash
   gdb ./my_program
   (gdb) run
   (gdb) bt  # Show backtrace when it crashes
   ```

2. **Check for NULL Pointers**:
   ```c
   ug_graph_t* graph = ug_create_graph();
   if (graph == NULL) {
       printf("Failed to create graph!\n");
       return 1;
   }
   ```

3. **Verify Node IDs**:
   ```c
   ug_node_id_t node = ug_create_string_node(graph, "test");
   if (node == UG_INVALID_ID) {
       printf("Failed to create node!\n");
       return 1;
   }
   ```

4. **Use Valgrind** (Linux/macOS):
   ```bash
   valgrind --leak-check=full ./my_program
   ```

#### Issue: "Memory leaks"
**Symptoms**:
Program uses increasing amounts of memory.

**Solutions**:
1. **Always Destroy Graphs**:
   ```c
   ug_graph_t* graph = ug_create_graph();
   // ... use graph ...
   ug_destroy_graph(graph);  // Don't forget this!
   ```

2. **Free Retrieved Strings**:
   ```c
   char* property = ug_get_node_property(graph, node, "name");
   if (property) {
       printf("Property: %s\n", property);
       free(property);  // Always free!
   }
   ```

3. **Check for Double-Free**:
   ```c
   char* data = ug_get_node_property(graph, node, "data");
   free(data);
   data = NULL;  // Prevent accidental reuse
   ```

### CLI Issues

#### Issue: "Command not recognized"
**Symptoms**:
```
ug> create_node alice
ERROR: Unknown command: create_node
```

**Solutions**:
1. **Use Correct Command Syntax**:
   ```bash
   ug> node alice        # Correct
   ug> create_node alice # Incorrect
   ```

2. **Check Available Commands**:
   ```bash
   ug> help
   ```

3. **Check Command Spelling**:
   ```bash
   ug> help node  # Get help for specific command
   ```

#### Issue: "Node not found"
**Symptoms**:
```
ug> edge alice bob
ERROR: Node 'alice' not found
```

**Solutions**:
1. **Create Nodes First**:
   ```bash
   ug> node alice
   ug> node bob
   ug> edge alice bob
   ```

2. **Check Node Names**:
   ```bash
   ug> nodes  # List all nodes
   ```

3. **Use Exact Names**:
   ```bash
   ug> node "Alice Smith"  # Use quotes for names with spaces
   ug> edge "Alice Smith" bob
   ```

### Performance Issues

#### Issue: "Graph operations are slow"
**Symptoms**:
Operations take a long time with large graphs.

**Solutions**:
1. **Enable Optimizations**:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

2. **Use Appropriate Data Structures**:
   ```c
   // For large graphs, consider using specialized graph types
   ug_graph_t* graph = ug_create_graph_with_type(UG_GRAPH_TYPE_LARGE_SCALE);
   ```

3. **Batch Operations**:
   ```c
   // Instead of many individual operations
   for (int i = 0; i < 1000; i++) {
       ug_create_string_node(graph, names[i]);
   }
   
   // Use batch operations when available
   ug_create_nodes_batch(graph, names, 1000);
   ```

4. **Enable Parallel Processing**:
   ```bash
   cmake .. -DUG_ENABLE_OPENMP=ON
   ```

#### Issue: "High memory usage"
**Symptoms**:
Program uses more memory than expected.

**Solutions**:
1. **Use Memory Pools**:
   ```c
   ug_memory_pool_t* pool = ug_create_memory_pool(1024 * 1024);
   ug_graph_t* graph = ug_create_graph_with_pool(pool);
   ```

2. **Clear Unused Data**:
   ```c
   // Remove nodes that are no longer needed
   ug_remove_node(graph, unused_node);
   
   // Compact the graph periodically
   ug_compact_graph(graph);
   ```

3. **Use Streaming for Large Datasets**:
   ```c
   // Process data in chunks instead of loading everything
   ug_graph_t* streaming_graph = ug_create_streaming_graph();
   ```

### Getting Help

#### Documentation
1. **Built-in Help**:
   ```bash
   ug_cli help
   ug_cli help node
   ```

2. **Manual Pages** (if installed):
   ```bash
   man universal_graph
   ```

3. **API Documentation**:
   - Look for `docs/` directory in installation
   - Check online documentation

#### Community Support
1. **GitHub Issues**: Report bugs and ask questions
2. **Stack Overflow**: Tag questions with `universal-graph-engine`
3. **Discord/Slack**: Join the community chat
4. **Mailing List**: Subscribe to updates and discussions

#### Professional Support
1. **Commercial License**: Includes professional support
2. **Consulting Services**: Custom development and optimization
3. **Training**: Workshops and courses

---

## FAQ

### General Questions

**Q: What makes Universal Graph Engine "universal"?**
A: It can store any data type, model any relationship pattern, scale to any size, and migrate to any programming language. Unlike traditional graph databases that have limitations on data types or relationship patterns, UGE is designed to handle infinite complexity.

**Q: Is it really infinite complexity?**
A: Practically infinite - limited only by available memory and computational resources. The engine supports:
- Unlimited node and edge types
- N-ary relationships (hypergraphs)
- Meta-relationships (relationships between relationships)
- Temporal evolution and quantum states
- Any custom data structures

**Q: How does it compare to Neo4j, Amazon Neptune, or other graph databases?**
A: Key differences:
- **Flexibility**: UGE can model relationships that others cannot (hypergraphs, quantum states, temporal evolution)
- **Data Types**: Truly universal - any data type, not just primitives
- **Language Support**: Native bindings for multiple languages with easy migration
- **Open Source**: Complete source code available for customization

### Technical Questions

**Q: What programming languages are supported?**
A: Currently supported:
- **C**: Native implementation
- **C++**: Zero-cost wrappers with RAII
- **Rust**: Memory-safe bindings
- **Python**: High-level interface (planned)
- **Go**: Concurrent-friendly bindings (planned)
- **JavaScript/Node.js**: Web integration (planned)

**Q: Can I use this in production?**
A: Yes, but consider:
- **Maturity**: This is version 1.0 - thoroughly test for your use case
- **Scale**: Tested up to millions of nodes, but verify for your scale
- **Support**: Community support is available, commercial support planned
- **Backup**: Implement proper backup strategies

**Q: How do I handle very large graphs (billions of nodes)?**
A: Several strategies:
1. **Distributed Mode**: Split across multiple machines
2. **Partitioning**: Divide graph into logical sections
3. **Streaming**: Process data in chunks
4. **Caching**: Use intelligent caching strategies
5. **SSD Storage**: Use fast storage for better performance

**Q: Is it thread-safe?**
A: Yes, with caveats:
- **Read Operations**: Multiple threads can read simultaneously
- **Write Operations**: Use locks or atomic operations
- **Graph-Level Locking**: Each graph has internal synchronization
- **Best Practice**: Use one graph per thread for writes, share for reads

### Performance Questions

**Q: How fast is it compared to other graph databases?**
A: Performance varies by use case:
- **Simple Operations**: Comparable to Neo4j for basic queries
- **Complex Operations**: Often faster due to optimized algorithms
- **Memory Usage**: Efficient memory layout, but depends on data types
- **Benchmark**: Run your specific workload for accurate comparison

**Q: Does it support GPU acceleration?**
A: Yes, optionally:
- **CUDA**: Enable with `-DUG_ENABLE_CUDA=ON`
- **OpenCL**: Enable with `-DUG_ENABLE_OPENCL=ON`
- **Use Cases**: Large-scale graph algorithms, machine learning
- **Requirements**: NVIDIA GPU for CUDA, compatible GPU for OpenCL

**Q: How much memory does it use?**
A: Depends on graph structure:
- **Base Overhead**: ~100MB for empty graph
- **Node Storage**: 50-200 bytes per node (depends on data)
- **Edge Storage**: 30-100 bytes per edge
- **Properties**: Variable, depends on data size
- **Optimization**: Use memory pools for better efficiency

### Usage Questions

**Q: Can I import data from other graph databases?**
A: Yes, several options:
1. **Built-in Importers**: 
   ```bash
   ug_cli import neo4j my_neo4j_export.cypher
   ug_cli import graphml my_graph.graphml
   ```
2. **Custom Scripts**: Write migration scripts using the API
3. **CSV Import**: Use standard CSV format
4. **JSON Import**: Import from JSON exports

**Q: How do I backup and restore graphs?**
A: Multiple approaches:
1. **Built-in Export**:
   ```bash
   ug_cli save my_backup.ug
   ug_cli load my_backup.ug
   ```
2. **Format-Specific Exports**:
   ```bash
   ug_cli export json full_backup.json
   ```
3. **Streaming Backup**: For large graphs, use streaming export
4. **Version Control**: Small graphs can be stored in Git

**Q: Can I run queries like SQL or Cypher?**
A: Query language support:
- **Native API**: Use C/C++/Rust/Python APIs directly
- **Graph Traversal**: Built-in traversal algorithms
- **Cypher-like**: Planned for future versions
- **Custom DSL**: You can build domain-specific query languages

### Integration Questions

**Q: How do I integrate with web applications?**
A: Several approaches:
1. **REST API**: Create HTTP endpoints using your language's web framework
2. **GraphQL**: Map graph operations to GraphQL resolvers
3. **WebSocket**: Real-time updates using streaming features
4. **JavaScript Binding**: Direct browser integration (planned)

**Q: Can I use it with Docker/Kubernetes?**
A: Yes:
1. **Docker Images**: Available for major platforms
2. **Kubernetes**: Supports horizontal scaling
3. **Helm Charts**: Available for easy deployment
4. **Configuration**: Environment variables for configuration

**Q: Does it work with machine learning frameworks?**
A: Yes, excellent integration:
1. **Python**: Direct integration with scikit-learn, PyTorch, TensorFlow
2. **Feature Extraction**: Convert graph properties to ML features
3. **Graph Neural Networks**: Native support for GNN frameworks
4. **Export Formats**: Compatible with NetworkX, DGL, PyTorch Geometric

### Licensing and Commercial Use

**Q: What's the license?**
A: MIT License:
- **Free**: Use in commercial and non-commercial projects
- **Open Source**: Full source code available
- **Modifications**: You can modify and redistribute
- **Attribution**: Must include copyright notice

**Q: Is commercial support available?**
A: Currently community-supported, commercial options planned:
- **Community**: GitHub issues, forums, documentation
- **Commercial**: Professional support, consulting, training (coming soon)
- **Enterprise**: Custom features, dedicated support (coming soon)

**Q: Can I contribute to the project?**
A: Absolutely! Contributions welcome:
1. **Bug Reports**: File issues on GitHub
2. **Feature Requests**: Propose new functionality
3. **Code Contributions**: Submit pull requests
4. **Documentation**: Improve docs and examples
5. **Testing**: Help test on different platforms

### Migration Questions

**Q: How hard is it to migrate from my current graph database?**
A: Difficulty varies:
- **Data Migration**: Usually straightforward with export/import
- **Query Migration**: May require rewriting queries
- **API Changes**: Application code changes needed
- **Testing**: Thorough testing required
- **Timeline**: Plan 2-12 weeks depending on complexity

**Q: Can I run both systems in parallel during migration?**
A: Yes, recommended approach:
1. **Dual Write**: Write to both systems during transition
2. **Read Preference**: Gradually shift reads to new system
3. **Validation**: Compare results between systems
4. **Rollback Plan**: Keep old system until confident

**Q: What if I need to go back to my old system?**
A: Migration is reversible:
1. **Export Data**: UGE can export to standard formats
2. **Schema Mapping**: Document schema differences
3. **Data Validation**: Verify data integrity after export
4. **Application Changes**: May need to revert code changes

This completes the comprehensive documentation. The Universal Graph Engine provides a powerful, flexible foundation for any application that needs to model and analyze connected data, from simple family trees to complex enterprise systems.