/**
 * Universal Graph Engine - Command Line Interface
 * 
 * Provides an interactive command-line interface for creating, manipulating,
 * and visualizing universal graphs with maximum ease of use.
 * 
 * Features:
 * - Interactive shell with tab completion
 * - Batch command processing
 * - Graph visualization and export
 * - Advanced node/edge manipulation
 * - Support for all graph types (simple, hyper, temporal, quantum)
 * 
 * Copyright (c) 2025 Universal Graph Engine Project
 * Licensed under MIT License
 */

#ifndef UG_CLI_H
#define UG_CLI_H

#include "universal_graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CLI Data Structures
 * ============================================================================ */

typedef enum {
    UG_CLI_MODE_INTERACTIVE,
    UG_CLI_MODE_BATCH,
    UG_CLI_MODE_SCRIPT
} ug_cli_mode_t;

typedef enum {
    UG_CLI_CMD_UNKNOWN = 0,
    UG_CLI_CMD_CREATE_NODE,
    UG_CLI_CMD_CREATE_EDGE,
    UG_CLI_CMD_DELETE_NODE,
    UG_CLI_CMD_DELETE_EDGE,
    UG_CLI_CMD_LIST_NODES,
    UG_CLI_CMD_LIST_EDGES,
    UG_CLI_CMD_SHOW_NODE,
    UG_CLI_CMD_SHOW_EDGE,
    UG_CLI_CMD_INSERT_NODE,
    UG_CLI_CMD_CREATE_TRIANGLE,
    UG_CLI_CMD_FIND_PATH,
    UG_CLI_CMD_NEIGHBORS,
    UG_CLI_CMD_STATS,
    UG_CLI_CMD_EXPORT,
    UG_CLI_CMD_IMPORT,
    UG_CLI_CMD_VISUALIZE,
    UG_CLI_CMD_HELP,
    UG_CLI_CMD_QUIT,
    UG_CLI_CMD_CLEAR,
    UG_CLI_CMD_SAVE,
    UG_CLI_CMD_LOAD,
    UG_CLI_CMD_SET_PROPERTY,
    UG_CLI_CMD_GET_PROPERTY,
    UG_CLI_CMD_CREATE_HYPEREDGE,
    UG_CLI_CMD_MAX
} ug_cli_command_t;

typedef struct {
    char* args[32];
    int argc;
    ug_cli_command_t command;
} ug_cli_parsed_command_t;

typedef struct {
    ug_graph_t* graph;
    ug_cli_mode_t mode;
    bool running;
    bool verbose;
    char* current_file;
    ug_node_id_t* node_name_map;
    char** node_names;
    size_t node_name_count;
    size_t node_name_capacity;
} ug_cli_context_t;

typedef struct {
    const char* name;
    const char* description;
    const char* usage;
    const char* examples;
    ug_cli_command_t command;
} ug_cli_command_info_t;

/* ============================================================================
 * CLI Core Functions
 * ============================================================================ */

/**
 * Initialize CLI context
 */
ug_cli_context_t* ug_cli_init(ug_cli_mode_t mode);

/**
 * Cleanup CLI context
 */
void ug_cli_cleanup(ug_cli_context_t* ctx);

/**
 * Main CLI loop
 */
int ug_cli_run(ug_cli_context_t* ctx);

/**
 * Execute a single command
 */
bool ug_cli_execute_command(ug_cli_context_t* ctx, const char* command_line);

/**
 * Parse command line into components
 */
ug_cli_parsed_command_t* ug_cli_parse_command(const char* command_line);

/**
 * Free parsed command
 */
void ug_cli_free_parsed_command(ug_cli_parsed_command_t* cmd);

/* ============================================================================
 * Command Handlers
 * ============================================================================ */

bool ug_cli_cmd_create_node(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_create_edge(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_delete_node(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_delete_edge(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_list_nodes(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_list_edges(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_show_node(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_show_edge(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_insert_node(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_create_triangle(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_find_path(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_neighbors(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_stats(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_export(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_import(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_visualize(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_help(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_quit(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_clear(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_save(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_load(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_set_property(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_get_property(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);
bool ug_cli_cmd_create_hyperedge(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Register a node name for easy reference
 */
bool ug_cli_register_node_name(ug_cli_context_t* ctx, const char* name, ug_node_id_t id);

/**
 * Resolve node name to ID
 */
ug_node_id_t ug_cli_resolve_node_name(ug_cli_context_t* ctx, const char* name);

/**
 * Get node name from ID
 */
const char* ug_cli_get_node_name(ug_cli_context_t* ctx, ug_node_id_t id);

/**
 * Parse value from string
 */
ug_universal_value_t* ug_cli_parse_value(const char* str);

/**
 * Print formatted output
 */
void ug_cli_printf(ug_cli_context_t* ctx, const char* format, ...);

/**
 * Print error message
 */
void ug_cli_error(ug_cli_context_t* ctx, const char* format, ...);

/**
 * Print warning message
 */
void ug_cli_warning(ug_cli_context_t* ctx, const char* format, ...);

/**
 * Print success message
 */
void ug_cli_success(ug_cli_context_t* ctx, const char* format, ...);

/**
 * Print graph visualization
 */
void ug_cli_print_graph_ascii(ug_cli_context_t* ctx);

/**
 * Read line with history and completion
 */
char* ug_cli_readline(const char* prompt);

/**
 * Add command to history
 */
void ug_cli_add_history(const char* command);

/**
 * Get command info
 */
const ug_cli_command_info_t* ug_cli_get_command_info(ug_cli_command_t cmd);

/**
 * Get all command info
 */
const ug_cli_command_info_t* ug_cli_get_all_commands(void);

/**
 * Get command count
 */
size_t ug_cli_get_command_count(void);

#ifdef __cplusplus
}
#endif

#endif /* UG_CLI_H */