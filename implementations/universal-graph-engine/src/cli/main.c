/**
 * Universal Graph Engine - CLI Main Entry Point
 * 
 * Main executable for the Universal Graph Engine command line interface.
 * Provides an interactive shell for creating and manipulating complex graphs.
 * 
 * Copyright (c) 2025 Universal Graph Engine Project
 * Licensed under MIT License
 */

#include "ug_cli.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#ifdef UG_PLATFORM_WINDOWS
#include <windows.h>
#else
#include <unistd.h>
#endif

/* Global CLI context for signal handling */
static ug_cli_context_t* g_cli_context = NULL;

/* Signal handler for graceful shutdown */
static void signal_handler(int sig) {
    if (g_cli_context) {
        printf("\n\nReceived signal %d. Shutting down gracefully...\n", sig);
        g_cli_context->running = false;
    } else {
        printf("\n\nReceived signal %d. Exiting immediately.\n", sig);
        exit(sig);
    }
}

/* Print usage information */
static void print_usage(const char* program_name) {
    printf("Universal Graph Engine CLI v1.0\n");
    printf("Usage: %s [options] [script_file]\n\n", program_name);
    printf("Options:\n");
    printf("  -h, --help     Show this help message\n");
    printf("  -v, --verbose  Enable verbose output\n");
    printf("  -q, --quiet    Disable verbose output\n");
    printf("  -i, --interactive  Force interactive mode (default)\n");
    printf("  -b, --batch    Enable batch mode (non-interactive)\n");
    printf("  -f, --file FILE    Execute commands from file\n");
    printf("  --version      Show version information\n\n");
    printf("Examples:\n");
    printf("  %s                    # Start interactive mode\n", program_name);
    printf("  %s -f script.ug       # Execute commands from file\n", program_name);
    printf("  %s --batch < input    # Process commands from stdin\n", program_name);
    printf("\nFor more information, visit: https://github.com/universal-graph-engine\n");
}

/* Print version information */
static void print_version(void) {
    printf("Universal Graph Engine CLI v1.0.0\n");
    printf("Built with Universal Graph Engine Core\n");
    printf("Copyright (c) 2025 Universal Graph Engine Project\n");
    printf("Licensed under MIT License\n");
}

/* Execute commands from file */
static int execute_script_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "ERROR: Cannot open script file '%s'\n", filename);
        return 1;
    }
    
    ug_cli_context_t* ctx = ug_cli_init(UG_CLI_MODE_SCRIPT);
    if (!ctx) {
        fprintf(stderr, "ERROR: Failed to initialize CLI context\n");
        fclose(file);
        return 1;
    }
    
    g_cli_context = ctx;
    
    char line[1024];
    int line_number = 0;
    bool success = true;
    
    printf("Executing script: %s\n", filename);
    
    while (fgets(line, sizeof(line), file) && ctx->running) {
        line_number++;
        
        // Remove newline
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }
        
        // Skip empty lines and comments
        char* trimmed = line;
        while (*trimmed == ' ' || *trimmed == '\t') trimmed++;
        
        if (*trimmed == '\0' || *trimmed == '#') {
            continue;
        }
        
        printf("[%d] %s\n", line_number, trimmed);
        
        if (!ug_cli_execute_command(ctx, trimmed)) {
            fprintf(stderr, "ERROR: Command failed at line %d: %s\n", line_number, trimmed);
            success = false;
            break;
        }
    }
    
    fclose(file);
    ug_cli_cleanup(ctx);
    g_cli_context = NULL;
    
    if (success) {
        printf("Script executed successfully.\n");
        return 0;
    } else {
        return 1;
    }
}

/* Execute commands from stdin in batch mode */
static int execute_batch_mode(void) {
    ug_cli_context_t* ctx = ug_cli_init(UG_CLI_MODE_BATCH);
    if (!ctx) {
        fprintf(stderr, "ERROR: Failed to initialize CLI context\n");
        return 1;
    }
    
    g_cli_context = ctx;
    
    char line[1024];
    int line_number = 0;
    bool success = true;
    
    while (fgets(line, sizeof(line), stdin) && ctx->running) {
        line_number++;
        
        // Remove newline
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }
        
        // Skip empty lines and comments
        char* trimmed = line;
        while (*trimmed == ' ' || *trimmed == '\t') trimmed++;
        
        if (*trimmed == '\0' || *trimmed == '#') {
            continue;
        }
        
        if (!ug_cli_execute_command(ctx, trimmed)) {
            fprintf(stderr, "ERROR: Command failed at line %d: %s\n", line_number, trimmed);
            success = false;
            break;
        }
    }
    
    ug_cli_cleanup(ctx);
    g_cli_context = NULL;
    
    return success ? 0 : 1;
}

/* Main entry point */
int main(int argc, char* argv[]) {
    ug_cli_mode_t mode = UG_CLI_MODE_INTERACTIVE;
    bool verbose = true;
    const char* script_file = NULL;
    
    /* Install signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--version") == 0) {
            print_version();
            return 0;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            verbose = false;
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0) {
            mode = UG_CLI_MODE_INTERACTIVE;
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--batch") == 0) {
            mode = UG_CLI_MODE_BATCH;
        } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--file") == 0) {
            if (i + 1 < argc) {
                script_file = argv[++i];
                mode = UG_CLI_MODE_SCRIPT;
            } else {
                fprintf(stderr, "ERROR: Option %s requires a filename\n", argv[i]);
                return 1;
            }
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "ERROR: Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        } else {
            /* Treat as script file */
            script_file = argv[i];
            mode = UG_CLI_MODE_SCRIPT;
        }
    }
    
    /* Execute based on mode */
    if (mode == UG_CLI_MODE_SCRIPT && script_file) {
        return execute_script_file(script_file);
    } else if (mode == UG_CLI_MODE_BATCH) {
        return execute_batch_mode();
    } else {
        /* Interactive mode */
        ug_cli_context_t* ctx = ug_cli_init(UG_CLI_MODE_INTERACTIVE);
        if (!ctx) {
            fprintf(stderr, "ERROR: Failed to initialize CLI context\n");
            return 1;
        }
        
        ctx->verbose = verbose;
        g_cli_context = ctx;
        
        int result = ug_cli_run(ctx);
        
        ug_cli_cleanup(ctx);
        g_cli_context = NULL;
        
        return result;
    }
}