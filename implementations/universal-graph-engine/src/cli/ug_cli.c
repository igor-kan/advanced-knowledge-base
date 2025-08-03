/**
 * Universal Graph Engine - Command Line Interface Implementation
 * 
 * Interactive CLI for creating and manipulating universal graphs with ease.
 * Supports advanced operations like inserting nodes between edges and creating triangles.
 * 
 * Copyright (c) 2025 Universal Graph Engine Project
 * Licensed under MIT License
 */

#include "ug_cli.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>

#ifdef UG_PLATFORM_WINDOWS
#include <windows.h>
#include <conio.h>
#else
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#endif

/* ============================================================================
 * Command Information Table
 * ============================================================================ */

static const ug_cli_command_info_t g_command_info[] = {
    {
        "node", "Create a new node",
        "node <name> [data] [type]",
        "node alice \"Alice Smith\"\nnode count 42 int\nnode temp 3.14 float",
        UG_CLI_CMD_CREATE_NODE
    },
    {
        "edge", "Create an edge between two nodes",
        "edge <from> <to> [type] [weight]",
        "edge alice bob KNOWS 1.0\nedge node1 node2 CONNECTED",
        UG_CLI_CMD_CREATE_EDGE
    },
    {
        "insert", "Insert a node between two connected nodes",
        "insert <between_from> <between_to> <new_node_name> [data]",
        "insert alice bob charlie \"Charlie Brown\"\ninsert n1 n2 middle",
        UG_CLI_CMD_INSERT_NODE
    },
    {
        "triangle", "Create a triangular connection (node connecting to two others)",
        "triangle <node1> <node2> <new_node> [data]",
        "triangle alice bob charlie \"Charlie connects both\"\ntriangle n1 n2 connector",
        UG_CLI_CMD_CREATE_TRIANGLE
    },
    {
        "hyperedge", "Create a hyperedge connecting multiple nodes",
        "hyperedge <type> <node1> <node2> [node3] ...",
        "hyperedge MEETING alice bob charlie\nhyperedge COLLABORATION n1 n2 n3 n4",
        UG_CLI_CMD_CREATE_HYPEREDGE
    },
    {
        "rm-node", "Delete a node and its edges",
        "rm-node <name>",
        "rm-node alice\nrm-node temp_node",
        UG_CLI_CMD_DELETE_NODE
    },
    {
        "rm-edge", "Delete an edge between nodes",
        "rm-edge <from> <to>",
        "rm-edge alice bob\nrm-edge n1 n2",
        UG_CLI_CMD_DELETE_EDGE
    },
    {
        "nodes", "List all nodes",
        "nodes [pattern]",
        "nodes\nnodes alice*\nnodes *temp*",
        UG_CLI_CMD_LIST_NODES
    },
    {
        "edges", "List all edges",
        "edges [from_pattern] [to_pattern]",
        "edges\nedges alice *\nedges * bob",
        UG_CLI_CMD_LIST_EDGES
    },
    {
        "show", "Show detailed information about a node or edge",
        "show <name> | show <from> <to>",
        "show alice\nshow alice bob",
        UG_CLI_CMD_SHOW_NODE
    },
    {
        "neighbors", "Show neighbors of a node",
        "neighbors <name> [depth]",
        "neighbors alice\nneighbors bob 2",
        UG_CLI_CMD_NEIGHBORS
    },
    {
        "path", "Find path between two nodes",
        "path <from> <to> [max_depth]",
        "path alice bob\npath n1 n5 3",
        UG_CLI_CMD_FIND_PATH
    },
    {
        "set", "Set property on a node",
        "set <node> <property> <value> [type]",
        "set alice age 25 int\nset bob city \"New York\"",
        UG_CLI_CMD_SET_PROPERTY
    },
    {
        "get", "Get property from a node",
        "get <node> <property>",
        "get alice age\nget bob city",
        UG_CLI_CMD_GET_PROPERTY
    },
    {
        "stats", "Show graph statistics",
        "stats",
        "stats",
        UG_CLI_CMD_STATS
    },
    {
        "export", "Export graph to file",
        "export <format> <filename>",
        "export dot graph.dot\nexport json graph.json",
        UG_CLI_CMD_EXPORT
    },
    {
        "import", "Import graph from file",
        "import <format> <filename>",
        "import dot graph.dot\nimport json graph.json",
        UG_CLI_CMD_IMPORT
    },
    {
        "viz", "Visualize graph in ASCII",
        "viz [layout]",
        "viz\nviz tree\nviz circular",
        UG_CLI_CMD_VISUALIZE
    },
    {
        "save", "Save current graph to file",
        "save <filename>",
        "save my_graph.ug",
        UG_CLI_CMD_SAVE
    },
    {
        "load", "Load graph from file",
        "load <filename>",
        "load my_graph.ug",
        UG_CLI_CMD_LOAD
    },
    {
        "clear", "Clear the current graph",
        "clear",
        "clear",
        UG_CLI_CMD_CLEAR
    },
    {
        "help", "Show help information",
        "help [command]",
        "help\nhelp node\nhelp insert",
        UG_CLI_CMD_HELP
    },
    {
        "quit", "Exit the CLI",
        "quit | exit | q",
        "quit",
        UG_CLI_CMD_QUIT
    }
};

static const size_t g_command_count = sizeof(g_command_info) / sizeof(g_command_info[0]);

/* ============================================================================
 * CLI Context Management
 * ============================================================================ */

ug_cli_context_t* ug_cli_init(ug_cli_mode_t mode) {
    ug_cli_context_t* ctx = (ug_cli_context_t*)calloc(1, sizeof(ug_cli_context_t));
    if (!ctx) return NULL;
    
    ctx->graph = ug_create_graph();
    if (!ctx->graph) {
        free(ctx);
        return NULL;
    }
    
    ctx->mode = mode;
    ctx->running = true;
    ctx->verbose = true;
    ctx->current_file = NULL;
    
    // Initialize node name mapping
    ctx->node_name_capacity = 128;
    ctx->node_names = (char**)calloc(ctx->node_name_capacity, sizeof(char*));
    ctx->node_name_map = (ug_node_id_t*)calloc(ctx->node_name_capacity, sizeof(ug_node_id_t));
    ctx->node_name_count = 0;
    
    return ctx;
}

void ug_cli_cleanup(ug_cli_context_t* ctx) {
    if (!ctx) return;
    
    if (ctx->graph) {
        ug_destroy_graph(ctx->graph);
    }
    
    if (ctx->current_file) {
        free(ctx->current_file);
    }
    
    // Free node names
    for (size_t i = 0; i < ctx->node_name_count; i++) {
        if (ctx->node_names[i]) {
            free(ctx->node_names[i]);
        }
    }
    
    if (ctx->node_names) {
        free(ctx->node_names);
    }
    
    if (ctx->node_name_map) {
        free(ctx->node_name_map);
    }
    
    free(ctx);
}

/* ============================================================================
 * Command Parsing
 * ============================================================================ */

static ug_cli_command_t ug_cli_parse_command_type(const char* cmd_str) {
    if (!cmd_str) return UG_CLI_CMD_UNKNOWN;
    
    // Convert to lowercase for comparison
    char lower_cmd[64];
    strncpy(lower_cmd, cmd_str, sizeof(lower_cmd) - 1);
    lower_cmd[sizeof(lower_cmd) - 1] = '\0';
    
    for (char* p = lower_cmd; *p; p++) {
        *p = tolower(*p);
    }
    
    // Check exact matches first
    for (size_t i = 0; i < g_command_count; i++) {
        if (strcmp(lower_cmd, g_command_info[i].name) == 0) {
            return g_command_info[i].command;
        }
    }
    
    // Check aliases
    if (strcmp(lower_cmd, "exit") == 0 || strcmp(lower_cmd, "q") == 0) {
        return UG_CLI_CMD_QUIT;
    }
    
    return UG_CLI_CMD_UNKNOWN;
}

ug_cli_parsed_command_t* ug_cli_parse_command(const char* command_line) {
    if (!command_line) return NULL;
    
    ug_cli_parsed_command_t* cmd = (ug_cli_parsed_command_t*)calloc(1, sizeof(ug_cli_parsed_command_t));
    if (!cmd) return NULL;
    
    // Create a copy of the command line for parsing
    size_t len = strlen(command_line);
    char* line_copy = (char*)malloc(len + 1);
    strcpy(line_copy, command_line);
    
    // Remove leading/trailing whitespace
    char* start = line_copy;
    while (isspace(*start)) start++;
    
    char* end = start + strlen(start) - 1;
    while (end > start && isspace(*end)) *end-- = '\0';
    
    if (*start == '\0') {
        free(line_copy);
        free(cmd);
        return NULL;
    }
    
    // Parse arguments
    cmd->argc = 0;
    char* token = strtok(start, " \t");
    
    while (token && cmd->argc < 31) {
        cmd->args[cmd->argc] = (char*)malloc(strlen(token) + 1);
        strcpy(cmd->args[cmd->argc], token);
        cmd->argc++;
        token = strtok(NULL, " \t");
    }
    
    cmd->args[cmd->argc] = NULL;
    
    // Determine command type
    if (cmd->argc > 0) {
        cmd->command = ug_cli_parse_command_type(cmd->args[0]);
    } else {
        cmd->command = UG_CLI_CMD_UNKNOWN;
    }
    
    free(line_copy);
    return cmd;
}

void ug_cli_free_parsed_command(ug_cli_parsed_command_t* cmd) {
    if (!cmd) return;
    
    for (int i = 0; i < cmd->argc; i++) {
        if (cmd->args[i]) {
            free(cmd->args[i]);
        }
    }
    
    free(cmd);
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

bool ug_cli_register_node_name(ug_cli_context_t* ctx, const char* name, ug_node_id_t id) {
    if (!ctx || !name) return false;
    
    // Check if we need to expand arrays
    if (ctx->node_name_count >= ctx->node_name_capacity) {
        size_t new_capacity = ctx->node_name_capacity * 2;
        
        char** new_names = (char**)realloc(ctx->node_names, new_capacity * sizeof(char*));
        ug_node_id_t* new_map = (ug_node_id_t*)realloc(ctx->node_name_map, new_capacity * sizeof(ug_node_id_t));
        
        if (!new_names || !new_map) {
            return false;
        }
        
        ctx->node_names = new_names;
        ctx->node_name_map = new_map;
        ctx->node_name_capacity = new_capacity;
    }
    
    // Add the mapping
    ctx->node_names[ctx->node_name_count] = (char*)malloc(strlen(name) + 1);
    strcpy(ctx->node_names[ctx->node_name_count], name);
    ctx->node_name_map[ctx->node_name_count] = id;
    ctx->node_name_count++;
    
    return true;
}

ug_node_id_t ug_cli_resolve_node_name(ug_cli_context_t* ctx, const char* name) {
    if (!ctx || !name) return UG_INVALID_ID;
    
    // Try to parse as numeric ID first
    char* endptr;
    unsigned long id = strtoul(name, &endptr, 10);
    if (*endptr == '\0' && id != 0) {
        return (ug_node_id_t)id;
    }
    
    // Search name mapping
    for (size_t i = 0; i < ctx->node_name_count; i++) {
        if (ctx->node_names[i] && strcmp(ctx->node_names[i], name) == 0) {
            return ctx->node_name_map[i];
        }
    }
    
    return UG_INVALID_ID;
}

const char* ug_cli_get_node_name(ug_cli_context_t* ctx, ug_node_id_t id) {
    if (!ctx) return NULL;
    
    for (size_t i = 0; i < ctx->node_name_count; i++) {
        if (ctx->node_name_map[i] == id) {
            return ctx->node_names[i];
        }
    }
    
    return NULL;
}

ug_universal_value_t* ug_cli_parse_value(const char* str) {
    if (!str) return NULL;
    
    ug_universal_value_t* value = (ug_universal_value_t*)calloc(1, sizeof(ug_universal_value_t));
    if (!value) return NULL;
    
    // Try to parse as number first
    char* endptr;
    
    // Try integer
    long long_val = strtol(str, &endptr, 10);
    if (*endptr == '\0') {
        value->type = UG_TYPE_INT;
        value->data.int_val = (int)long_val;
        return value;
    }
    
    // Try float
    double double_val = strtod(str, &endptr);
    if (*endptr == '\0') {
        value->type = UG_TYPE_DOUBLE;
        value->data.double_val = double_val;
        return value;
    }
    
    // Try boolean
    if (strcasecmp(str, "true") == 0 || strcasecmp(str, "1") == 0) {
        value->type = UG_TYPE_BOOL;
        value->data.bool_val = true;
        return value;
    }
    
    if (strcasecmp(str, "false") == 0 || strcasecmp(str, "0") == 0) {
        value->type = UG_TYPE_BOOL;
        value->data.bool_val = false;
        return value;
    }
    
    // Default to string
    value->type = UG_TYPE_STRING;
    size_t len = strlen(str);
    
    // Remove quotes if present
    if (len >= 2 && str[0] == '"' && str[len-1] == '"') {
        value->data.string_val = (char*)malloc(len - 1);
        strncpy(value->data.string_val, str + 1, len - 2);
        value->data.string_val[len - 2] = '\0';
    } else {
        value->data.string_val = (char*)malloc(len + 1);
        strcpy(value->data.string_val, str);
    }
    
    return value;
}

void ug_cli_printf(ug_cli_context_t* ctx, const char* format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    fflush(stdout);
}

void ug_cli_error(ug_cli_context_t* ctx, const char* format, ...) {
    printf("ERROR: ");
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
    fflush(stdout);
}

void ug_cli_warning(ug_cli_context_t* ctx, const char* format, ...) {
    printf("WARNING: ");
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
    fflush(stdout);
}

void ug_cli_success(ug_cli_context_t* ctx, const char* format, ...) {
    printf("SUCCESS: ");
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
    fflush(stdout);
}

char* ug_cli_readline(const char* prompt) {
    printf("%s", prompt);
    fflush(stdout);
    
    char* line = (char*)malloc(1024);
    if (!line) return NULL;
    
    if (fgets(line, 1024, stdin) == NULL) {
        free(line);
        return NULL;
    }
    
    // Remove newline
    size_t len = strlen(line);
    if (len > 0 && line[len-1] == '\n') {
        line[len-1] = '\0';
    }
    
    return line;
}

void ug_cli_add_history(const char* command) {
    // Simple implementation - could be enhanced with readline library
    (void)command; // Suppress unused parameter warning
}

const ug_cli_command_info_t* ug_cli_get_command_info(ug_cli_command_t cmd) {
    for (size_t i = 0; i < g_command_count; i++) {
        if (g_command_info[i].command == cmd) {
            return &g_command_info[i];
        }
    }
    return NULL;
}

const ug_cli_command_info_t* ug_cli_get_all_commands(void) {
    return g_command_info;
}

size_t ug_cli_get_command_count(void) {
    return g_command_count;
}

/* ============================================================================
 * Command Handlers
 * ============================================================================ */

bool ug_cli_cmd_create_node(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    if (cmd->argc < 2) {
        ug_cli_error(ctx, "Usage: node <name> [data] [type]");
        return false;
    }
    
    const char* name = cmd->args[1];
    const char* data_str = (cmd->argc > 2) ? cmd->args[2] : name;
    
    ug_universal_value_t* value = ug_cli_parse_value(data_str);
    if (!value) {
        ug_cli_error(ctx, "Failed to parse node data");
        return false;
    }
    
    ug_node_id_t node_id = ug_create_node(ctx->graph, value->type, &value->data);
    
    if (node_id == UG_INVALID_ID) {
        ug_cli_error(ctx, "Failed to create node");
        // Free value data if needed
        if (value->type == UG_TYPE_STRING && value->data.string_val) {
            free(value->data.string_val);
        }
        free(value);
        return false;
    }
    
    if (!ug_cli_register_node_name(ctx, name, node_id)) {
        ug_cli_warning(ctx, "Created node %lu but failed to register name '%s'", node_id, name);
    } else {
        ug_cli_success(ctx, "Created node '%s' with ID %lu", name, node_id);
    }
    
    // Free value data if needed
    if (value->type == UG_TYPE_STRING && value->data.string_val) {
        free(value->data.string_val);
    }
    free(value);
    
    return true;
}

bool ug_cli_cmd_create_edge(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    if (cmd->argc < 3) {
        ug_cli_error(ctx, "Usage: edge <from> <to> [type] [weight]");
        return false;
    }
    
    ug_node_id_t from_id = ug_cli_resolve_node_name(ctx, cmd->args[1]);
    ug_node_id_t to_id = ug_cli_resolve_node_name(ctx, cmd->args[2]);
    
    if (from_id == UG_INVALID_ID) {
        ug_cli_error(ctx, "Node '%s' not found", cmd->args[1]);
        return false;
    }
    
    if (to_id == UG_INVALID_ID) {
        ug_cli_error(ctx, "Node '%s' not found", cmd->args[2]);
        return false;
    }
    
    const char* edge_type = (cmd->argc > 3) ? cmd->args[3] : "CONNECTED";
    ug_weight_t weight = (cmd->argc > 4) ? atof(cmd->args[4]) : 1.0;
    
    ug_relationship_id_t rel_id = ug_create_edge(ctx->graph, from_id, to_id, edge_type, weight);
    
    if (rel_id == UG_INVALID_ID) {
        ug_cli_error(ctx, "Failed to create edge");
        return false;
    }
    
    ug_cli_success(ctx, "Created edge %s -> %s (type: %s, weight: %.2f, ID: %lu)", 
                   cmd->args[1], cmd->args[2], edge_type, weight, rel_id);
    
    return true;
}

bool ug_cli_cmd_insert_node(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    if (cmd->argc < 4) {
        ug_cli_error(ctx, "Usage: insert <from> <to> <new_node_name> [data]");
        return false;
    }
    
    ug_node_id_t from_id = ug_cli_resolve_node_name(ctx, cmd->args[1]);
    ug_node_id_t to_id = ug_cli_resolve_node_name(ctx, cmd->args[2]);
    
    if (from_id == UG_INVALID_ID) {
        ug_cli_error(ctx, "Node '%s' not found", cmd->args[1]);
        return false;
    }
    
    if (to_id == UG_INVALID_ID) {
        ug_cli_error(ctx, "Node '%s' not found", cmd->args[2]);
        return false;
    }
    
    // Check if edge exists between from and to
    // This is a simplified check - in a full implementation we'd properly search for the edge
    const char* new_name = cmd->args[3];
    const char* data_str = (cmd->argc > 4) ? cmd->args[4] : new_name;
    
    // Create the new node
    ug_universal_value_t* value = ug_cli_parse_value(data_str);
    if (!value) {
        ug_cli_error(ctx, "Failed to parse node data");
        return false;
    }
    
    ug_node_id_t new_node_id = ug_create_node(ctx->graph, value->type, &value->data);
    
    if (new_node_id == UG_INVALID_ID) {
        ug_cli_error(ctx, "Failed to create intermediate node");
        if (value->type == UG_TYPE_STRING && value->data.string_val) {
            free(value->data.string_val);
        }
        free(value);
        return false;
    }
    
    // Register the new node name
    ug_cli_register_node_name(ctx, new_name, new_node_id);
    
    // Create edges: from -> new_node and new_node -> to
    ug_relationship_id_t rel1 = ug_create_edge(ctx->graph, from_id, new_node_id, "CONNECTED", 1.0);
    ug_relationship_id_t rel2 = ug_create_edge(ctx->graph, new_node_id, to_id, "CONNECTED", 1.0);
    
    if (rel1 == UG_INVALID_ID || rel2 == UG_INVALID_ID) {
        ug_cli_error(ctx, "Failed to create connecting edges");
    } else {
        ug_cli_success(ctx, "Inserted node '%s' between '%s' and '%s'", 
                       new_name, cmd->args[1], cmd->args[2]);
    }
    
    // Note: In a full implementation, we'd also remove the original direct edge
    // between from and to if it existed
    
    // Free value data if needed
    if (value->type == UG_TYPE_STRING && value->data.string_val) {
        free(value->data.string_val);
    }
    free(value);
    
    return true;
}

bool ug_cli_cmd_create_triangle(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    if (cmd->argc < 4) {
        ug_cli_error(ctx, "Usage: triangle <node1> <node2> <new_node> [data]");
        return false;
    }
    
    ug_node_id_t node1_id = ug_cli_resolve_node_name(ctx, cmd->args[1]);
    ug_node_id_t node2_id = ug_cli_resolve_node_name(ctx, cmd->args[2]);
    
    if (node1_id == UG_INVALID_ID) {
        ug_cli_error(ctx, "Node '%s' not found", cmd->args[1]);
        return false;
    }
    
    if (node2_id == UG_INVALID_ID) {
        ug_cli_error(ctx, "Node '%s' not found", cmd->args[2]);
        return false;
    }
    
    const char* new_name = cmd->args[3];
    const char* data_str = (cmd->argc > 4) ? cmd->args[4] : new_name;
    
    // Create the new node
    ug_universal_value_t* value = ug_cli_parse_value(data_str);
    if (!value) {
        ug_cli_error(ctx, "Failed to parse node data");
        return false;
    }
    
    ug_node_id_t new_node_id = ug_create_node(ctx->graph, value->type, &value->data);
    
    if (new_node_id == UG_INVALID_ID) {
        ug_cli_error(ctx, "Failed to create new node");
        if (value->type == UG_TYPE_STRING && value->data.string_val) {
            free(value->data.string_val);
        }
        free(value);
        return false;
    }
    
    // Register the new node name
    ug_cli_register_node_name(ctx, new_name, new_node_id);
    
    // Create edges: new_node -> node1 and new_node -> node2
    ug_relationship_id_t rel1 = ug_create_edge(ctx->graph, new_node_id, node1_id, "CONNECTS", 1.0);
    ug_relationship_id_t rel2 = ug_create_edge(ctx->graph, new_node_id, node2_id, "CONNECTS", 1.0);
    
    if (rel1 == UG_INVALID_ID || rel2 == UG_INVALID_ID) {
        ug_cli_error(ctx, "Failed to create triangle edges");
    } else {
        ug_cli_success(ctx, "Created triangle: '%s' connects to both '%s' and '%s'", 
                       new_name, cmd->args[1], cmd->args[2]);
    }
    
    // Free value data if needed
    if (value->type == UG_TYPE_STRING && value->data.string_val) {
        free(value->data.string_val);
    }
    free(value);
    
    return true;
}

bool ug_cli_cmd_list_nodes(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_printf(ctx, "\nNodes in graph (%zu total):\n", ctx->node_name_count);
    ug_cli_printf(ctx, "%-20s %-10s %s\n", "Name", "ID", "Data");
    ug_cli_printf(ctx, "%-20s %-10s %s\n", "----", "--", "----");
    
    for (size_t i = 0; i < ctx->node_name_count; i++) {
        const char* name = ctx->node_names[i];
        ug_node_id_t id = ctx->node_name_map[i];
        
        // Get node data (simplified - in full implementation we'd properly retrieve and format it)
        ug_cli_printf(ctx, "%-20s %-10lu %s\n", name, id, "<data>");
    }
    
    return true;
}

bool ug_cli_cmd_stats(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    size_t node_count = ug_get_node_count(ctx->graph);
    size_t relationship_count = ug_get_relationship_count(ctx->graph);
    
    ug_cli_printf(ctx, "\nGraph Statistics:\n");
    ug_cli_printf(ctx, "  Nodes: %zu\n", node_count);
    ug_cli_printf(ctx, "  Relationships: %zu\n", relationship_count);
    ug_cli_printf(ctx, "  Named nodes: %zu\n", ctx->node_name_count);
    
    if (node_count > 0) {
        double density = (double)(relationship_count * 2) / (node_count * (node_count - 1));
        ug_cli_printf(ctx, "  Graph density: %.4f\n", density);
    }
    
    return true;
}

bool ug_cli_cmd_help(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    if (cmd->argc > 1) {
        // Show help for specific command
        const char* cmd_name = cmd->args[1];
        
        for (size_t i = 0; i < g_command_count; i++) {
            if (strcmp(g_command_info[i].name, cmd_name) == 0) {
                const ug_cli_command_info_t* info = &g_command_info[i];
                ug_cli_printf(ctx, "\n%s - %s\n", info->name, info->description);
                ug_cli_printf(ctx, "Usage: %s\n", info->usage);
                ug_cli_printf(ctx, "\nExamples:\n%s\n", info->examples);
                return true;
            }
        }
        
        ug_cli_error(ctx, "Unknown command: %s", cmd_name);
        return false;
    }
    
    // Show general help
    ug_cli_printf(ctx, "\nUniversal Graph Engine CLI\n");
    ug_cli_printf(ctx, "==========================\n\n");
    ug_cli_printf(ctx, "Available commands:\n\n");
    
    for (size_t i = 0; i < g_command_count; i++) {
        const ug_cli_command_info_t* info = &g_command_info[i];
        ug_cli_printf(ctx, "  %-12s - %s\n", info->name, info->description);
    }
    
    ug_cli_printf(ctx, "\nTip: Use 'help <command>' for detailed information about a specific command.\n");
    ug_cli_printf(ctx, "Example: help insert\n\n");
    
    return true;
}

bool ug_cli_cmd_quit(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ctx->running = false;
    ug_cli_printf(ctx, "Goodbye!\n");
    return true;
}

// Placeholder implementations for remaining commands
bool ug_cli_cmd_delete_node(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_error(ctx, "Command not yet implemented");
    return false;
}

bool ug_cli_cmd_delete_edge(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_error(ctx, "Command not yet implemented");
    return false;
}

bool ug_cli_cmd_list_edges(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_error(ctx, "Command not yet implemented");
    return false;
}

bool ug_cli_cmd_show_node(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_error(ctx, "Command not yet implemented");
    return false;
}

bool ug_cli_cmd_show_edge(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_error(ctx, "Command not yet implemented");
    return false;
}

bool ug_cli_cmd_find_path(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_error(ctx, "Command not yet implemented");
    return false;
}

bool ug_cli_cmd_neighbors(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_error(ctx, "Command not yet implemented");
    return false;
}

bool ug_cli_cmd_export(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_error(ctx, "Command not yet implemented");
    return false;
}

bool ug_cli_cmd_import(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_error(ctx, "Command not yet implemented");
    return false;
}

bool ug_cli_cmd_visualize(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_error(ctx, "Command not yet implemented");
    return false;
}

bool ug_cli_cmd_clear(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_error(ctx, "Command not yet implemented");
    return false;
}

bool ug_cli_cmd_save(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_error(ctx, "Command not yet implemented");
    return false;
}

bool ug_cli_cmd_load(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_error(ctx, "Command not yet implemented");
    return false;
}

bool ug_cli_cmd_set_property(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_error(ctx, "Command not yet implemented");
    return false;
}

bool ug_cli_cmd_get_property(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_error(ctx, "Command not yet implemented");
    return false;
}

bool ug_cli_cmd_create_hyperedge(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_error(ctx, "Command not yet implemented");
    return false;
}

/* ============================================================================
 * Main Execution
 * ============================================================================ */

bool ug_cli_execute_command(ug_cli_context_t* ctx, const char* command_line) {
    if (!ctx || !command_line) return false;
    
    ug_cli_parsed_command_t* cmd = ug_cli_parse_command(command_line);
    if (!cmd) return true; // Empty command is not an error
    
    bool result = true;
    
    switch (cmd->command) {
        case UG_CLI_CMD_CREATE_NODE:
            result = ug_cli_cmd_create_node(ctx, cmd);
            break;
        case UG_CLI_CMD_CREATE_EDGE:
            result = ug_cli_cmd_create_edge(ctx, cmd);
            break;
        case UG_CLI_CMD_INSERT_NODE:
            result = ug_cli_cmd_insert_node(ctx, cmd);
            break;
        case UG_CLI_CMD_CREATE_TRIANGLE:
            result = ug_cli_cmd_create_triangle(ctx, cmd);
            break;
        case UG_CLI_CMD_LIST_NODES:
            result = ug_cli_cmd_list_nodes(ctx, cmd);
            break;
        case UG_CLI_CMD_STATS:
            result = ug_cli_cmd_stats(ctx, cmd);
            break;
        case UG_CLI_CMD_HELP:
            result = ug_cli_cmd_help(ctx, cmd);
            break;
        case UG_CLI_CMD_QUIT:
            result = ug_cli_cmd_quit(ctx, cmd);
            break;
        case UG_CLI_CMD_UNKNOWN:
            ug_cli_error(ctx, "Unknown command: %s. Type 'help' for available commands.", cmd->args[0]);
            result = false;
            break;
        default:
            ug_cli_error(ctx, "Command not yet implemented: %s", cmd->args[0]);
            result = false;
            break;
    }
    
    ug_cli_free_parsed_command(cmd);
    return result;
}

int ug_cli_run(ug_cli_context_t* ctx) {
    if (!ctx) return 1;
    
    ug_cli_printf(ctx, "Universal Graph Engine CLI v1.0\n");
    ug_cli_printf(ctx, "Type 'help' for available commands, 'quit' to exit.\n\n");
    
    while (ctx->running) {
        char* line = ug_cli_readline("ug> ");
        if (!line) {
            // EOF reached
            ctx->running = false;
            break;
        }
        
        if (strlen(line) > 0) {
            ug_cli_add_history(line);
            ug_cli_execute_command(ctx, line);
        }
        
        free(line);
    }
    
    return 0;
}