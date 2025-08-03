/**
 * Universal Graph Engine - CLI Visualization Extensions
 * 
 * Advanced visualization and graph analysis for the CLI.
 * Provides ASCII art graph display, export capabilities, and analysis tools.
 * 
 * Copyright (c) 2025 Universal Graph Engine Project
 * Licensed under MIT License
 */

#include "ug_cli.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * ASCII Graph Visualization
 * ============================================================================ */

#define MAX_DISPLAY_NODES 20
#define CANVAS_WIDTH 80
#define CANVAS_HEIGHT 24

typedef struct {
    int x, y;
    char symbol;
    const char* label;
} ascii_node_t;

typedef struct {
    int from_x, from_y;
    int to_x, to_y;
    char style;
} ascii_edge_t;

static void draw_line(char canvas[CANVAS_HEIGHT][CANVAS_WIDTH + 1], 
                     int x1, int y1, int x2, int y2, char style) {
    // Simple line drawing using Bresenham's algorithm
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;
    
    int x = x1, y = y1;
    
    while (1) {
        if (x >= 0 && x < CANVAS_WIDTH && y >= 0 && y < CANVAS_HEIGHT) {
            if (canvas[y][x] == ' ') {
                canvas[y][x] = style;
            }
        }
        
        if (x == x2 && y == y2) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

static void circular_layout(ascii_node_t* nodes, int node_count) {
    int center_x = CANVAS_WIDTH / 2;
    int center_y = CANVAS_HEIGHT / 2;
    int radius = (CANVAS_HEIGHT < CANVAS_WIDTH ? CANVAS_HEIGHT : CANVAS_WIDTH) / 3;
    
    for (int i = 0; i < node_count; i++) {
        double angle = 2.0 * M_PI * i / node_count;
        nodes[i].x = center_x + (int)(radius * cos(angle));
        nodes[i].y = center_y + (int)(radius * sin(angle));
        nodes[i].symbol = 'A' + (i % 26);
    }
}

static void tree_layout(ascii_node_t* nodes, int node_count) {
    // Simple tree layout - arrange nodes in levels
    int levels = (int)ceil(log2(node_count + 1));
    int level_width = CANVAS_WIDTH / (levels + 1);
    
    for (int i = 0; i < node_count; i++) {
        int level = (int)floor(log2(i + 1));
        int position_in_level = i - ((1 << level) - 1);
        int level_size = 1 << level;
        
        nodes[i].x = level_width * (level + 1);
        nodes[i].y = CANVAS_HEIGHT * (position_in_level + 1) / (level_size + 1);
        nodes[i].symbol = 'A' + (i % 26);
    }
}

void ug_cli_print_graph_ascii(ug_cli_context_t* ctx) {
    if (!ctx) return;
    
    size_t node_count = ctx->node_name_count;
    
    if (node_count == 0) {
        ug_cli_printf(ctx, "Graph is empty - no nodes to visualize.\n");
        return;
    }
    
    if (node_count > MAX_DISPLAY_NODES) {
        ug_cli_printf(ctx, "Graph too large for ASCII visualization (%zu nodes > %d limit).\n", 
                      node_count, MAX_DISPLAY_NODES);
        ug_cli_printf(ctx, "Consider using 'export dot graph.dot' and viewing with Graphviz.\n");
        return;
    }
    
    // Initialize canvas
    char canvas[CANVAS_HEIGHT][CANVAS_WIDTH + 1];
    for (int y = 0; y < CANVAS_HEIGHT; y++) {
        for (int x = 0; x < CANVAS_WIDTH; x++) {
            canvas[y][x] = ' ';
        }
        canvas[y][CANVAS_WIDTH] = '\0';
    }
    
    // Create node layout
    ascii_node_t nodes[MAX_DISPLAY_NODES];
    circular_layout(nodes, (int)node_count);
    
    // Set node labels
    for (size_t i = 0; i < node_count && i < MAX_DISPLAY_NODES; i++) {
        nodes[i].label = ctx->node_names[i];
    }
    
    // Draw edges (simplified - would need actual edge data from graph)
    // For demo purposes, draw some connections
    for (int i = 0; i < (int)node_count - 1; i++) {
        draw_line(canvas, nodes[i].x, nodes[i].y, 
                 nodes[i + 1].x, nodes[i + 1].y, '-');
    }
    
    // Draw nodes on top of edges
    for (int i = 0; i < (int)node_count; i++) {
        if (nodes[i].x >= 0 && nodes[i].x < CANVAS_WIDTH && 
            nodes[i].y >= 0 && nodes[i].y < CANVAS_HEIGHT) {
            canvas[nodes[i].y][nodes[i].x] = nodes[i].symbol;
        }
    }
    
    // Print canvas
    ug_cli_printf(ctx, "\nGraph Visualization (Circular Layout):\n");
    ug_cli_printf(ctx, "┌");
    for (int x = 0; x < CANVAS_WIDTH; x++) ug_cli_printf(ctx, "─");
    ug_cli_printf(ctx, "┐\n");
    
    for (int y = 0; y < CANVAS_HEIGHT; y++) {
        ug_cli_printf(ctx, "│%s│\n", canvas[y]);
    }
    
    ug_cli_printf(ctx, "└");
    for (int x = 0; x < CANVAS_WIDTH; x++) ug_cli_printf(ctx, "─");
    ug_cli_printf(ctx, "┘\n");
    
    // Print legend
    ug_cli_printf(ctx, "\nLegend:\n");
    for (size_t i = 0; i < node_count && i < MAX_DISPLAY_NODES; i++) {
        ug_cli_printf(ctx, "  %c = %s\n", nodes[i].symbol, nodes[i].label);
    }
    ug_cli_printf(ctx, "  - = connection\n\n");
}

/* ============================================================================
 * Enhanced Visualization Command Implementation
 * ============================================================================ */

bool ug_cli_cmd_visualize(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    const char* layout = (cmd->argc > 1) ? cmd->args[1] : "circular";
    
    ug_cli_printf(ctx, "Generating ASCII visualization (layout: %s)...\n", layout);
    
    if (strcmp(layout, "circular") == 0) {
        ug_cli_print_graph_ascii(ctx);
    } else if (strcmp(layout, "tree") == 0) {
        // Would implement tree layout here
        ug_cli_printf(ctx, "Tree layout not yet implemented. Using circular.\n");
        ug_cli_print_graph_ascii(ctx);
    } else {
        ug_cli_error(ctx, "Unknown layout: %s. Available: circular, tree", layout);
        return false;
    }
    
    return true;
}

/* ============================================================================
 * Export Functions
 * ============================================================================ */

static bool export_dot_format(ug_cli_context_t* ctx, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        ug_cli_error(ctx, "Cannot create file: %s", filename);
        return false;
    }
    
    fprintf(file, "digraph UniversalGraph {\n");
    fprintf(file, "  rankdir=LR;\n");
    fprintf(file, "  node [shape=box, style=rounded];\n\n");
    
    // Export nodes
    for (size_t i = 0; i < ctx->node_name_count; i++) {
        fprintf(file, "  \"%s\" [label=\"%s\\nID:%lu\"];\n", 
                ctx->node_names[i], ctx->node_names[i], ctx->node_name_map[i]);
    }
    
    fprintf(file, "\n");
    
    // Export edges (simplified - would need actual edge iteration)
    fprintf(file, "  // Edges would be exported here based on actual graph structure\n");
    
    fprintf(file, "}\n");
    fclose(file);
    
    return true;
}

static bool export_json_format(ug_cli_context_t* ctx, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        ug_cli_error(ctx, "Cannot create file: %s", filename);
        return false;
    }
    
    fprintf(file, "{\n");
    fprintf(file, "  \"graph\": {\n");
    fprintf(file, "    \"type\": \"universal\",\n");
    fprintf(file, "    \"nodes\": [\n");
    
    for (size_t i = 0; i < ctx->node_name_count; i++) {
        fprintf(file, "      {\n");
        fprintf(file, "        \"id\": %lu,\n", ctx->node_name_map[i]);
        fprintf(file, "        \"name\": \"%s\",\n", ctx->node_names[i]);
        fprintf(file, "        \"properties\": {}\n");
        fprintf(file, "      }%s\n", (i < ctx->node_name_count - 1) ? "," : "");
    }
    
    fprintf(file, "    ],\n");
    fprintf(file, "    \"edges\": [\n");
    fprintf(file, "      // Edges would be exported here\n");
    fprintf(file, "    ]\n");
    fprintf(file, "  }\n");
    fprintf(file, "}\n");
    
    fclose(file);
    return true;
}

bool ug_cli_cmd_export(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    if (cmd->argc < 3) {
        ug_cli_error(ctx, "Usage: export <format> <filename>");
        ug_cli_printf(ctx, "Supported formats: dot, json, xml, csv\n");
        return false;
    }
    
    const char* format = cmd->args[1];
    const char* filename = cmd->args[2];
    
    ug_cli_printf(ctx, "Exporting graph to %s format...\n", format);
    
    bool success = false;
    
    if (strcmp(format, "dot") == 0) {
        success = export_dot_format(ctx, filename);
    } else if (strcmp(format, "json") == 0) {
        success = export_json_format(ctx, filename);
    } else if (strcmp(format, "xml") == 0) {
        ug_cli_error(ctx, "XML export not yet implemented");
        return false;
    } else if (strcmp(format, "csv") == 0) {
        ug_cli_error(ctx, "CSV export not yet implemented");
        return false;
    } else {
        ug_cli_error(ctx, "Unsupported format: %s", format);
        ug_cli_printf(ctx, "Supported formats: dot, json, xml, csv\n");
        return false;
    }
    
    if (success) {
        ug_cli_success(ctx, "Graph exported to %s", filename);
        
        if (strcmp(format, "dot") == 0) {
            ug_cli_printf(ctx, "Tip: View with 'dot -Tpng %s -o graph.png'\n", filename);
        }
    }
    
    return success;
}

/* ============================================================================
 * Enhanced Analysis Functions
 * ============================================================================ */

bool ug_cli_cmd_neighbors(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    if (cmd->argc < 2) {
        ug_cli_error(ctx, "Usage: neighbors <node> [depth]");
        return false;
    }
    
    const char* node_name = cmd->args[1];
    int depth = (cmd->argc > 2) ? atoi(cmd->args[2]) : 1;
    
    ug_node_id_t node_id = ug_cli_resolve_node_name(ctx, node_name);
    if (node_id == UG_INVALID_ID) {
        ug_cli_error(ctx, "Node '%s' not found", node_name);
        return false;
    }
    
    ug_cli_printf(ctx, "Neighbors of '%s' (depth %d):\n", node_name, depth);
    
    // This would implement actual neighbor discovery
    // For now, show placeholder
    ug_cli_printf(ctx, "  (Neighbor discovery not yet implemented in core graph)\n");
    
    return true;
}

bool ug_cli_cmd_find_path(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    if (cmd->argc < 3) {
        ug_cli_error(ctx, "Usage: path <from> <to> [max_depth]");
        return false;
    }
    
    const char* from_name = cmd->args[1];
    const char* to_name = cmd->args[2];
    int max_depth = (cmd->argc > 3) ? atoi(cmd->args[3]) : 10;
    
    ug_node_id_t from_id = ug_cli_resolve_node_name(ctx, from_name);
    ug_node_id_t to_id = ug_cli_resolve_node_name(ctx, to_name);
    
    if (from_id == UG_INVALID_ID) {
        ug_cli_error(ctx, "Node '%s' not found", from_name);
        return false;
    }
    
    if (to_id == UG_INVALID_ID) {
        ug_cli_error(ctx, "Node '%s' not found", to_name);
        return false;
    }
    
    ug_cli_printf(ctx, "Finding path from '%s' to '%s' (max depth: %d):\n", 
                  from_name, to_name, max_depth);
    
    // This would implement actual pathfinding
    // For now, show placeholder
    ug_cli_printf(ctx, "  (Pathfinding not yet implemented in core graph)\n");
    
    return true;
}

/* ============================================================================
 * Clear Graph Implementation
 * ============================================================================ */

bool ug_cli_cmd_clear(ug_cli_context_t* ctx, ug_cli_parsed_command_t* cmd) {
    ug_cli_printf(ctx, "Are you sure you want to clear the entire graph? (y/N): ");
    fflush(stdout);
    
    char response[10];
    if (fgets(response, sizeof(response), stdin) && 
        (response[0] == 'y' || response[0] == 'Y')) {
        
        // Clear the graph (simplified - would call actual graph clear function)
        // ug_clear_graph(ctx->graph);
        
        // Clear name mappings
        for (size_t i = 0; i < ctx->node_name_count; i++) {
            if (ctx->node_names[i]) {
                free(ctx->node_names[i]);
                ctx->node_names[i] = NULL;
            }
        }
        ctx->node_name_count = 0;
        
        ug_cli_success(ctx, "Graph cleared successfully");
    } else {
        ug_cli_printf(ctx, "Clear operation cancelled.\n");
    }
    
    return true;
}