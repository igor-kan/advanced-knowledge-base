# Universal Graph Engine - Examples
cmake_minimum_required(VERSION 3.15)

# Include header directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Link against the universal graph library
if(UG_BUILD_SHARED)
    set(UG_LIBRARY universal_graph)
else()
    set(UG_LIBRARY universal_graph_static)
endif()

# Example 1: Infinite Complexity Demonstration
add_executable(infinite_complexity_demo infinite_complexity_demo.c)
target_link_libraries(infinite_complexity_demo ${UG_LIBRARY} ${UG_PLATFORM_LIBS})
target_compile_options(infinite_complexity_demo PRIVATE ${UG_C_FLAGS})

# Example 2: Statistical Knowledge Graph
add_executable(statistical_knowledge_graph statistical_knowledge_graph.c)
target_link_libraries(statistical_knowledge_graph ${UG_LIBRARY} ${UG_PLATFORM_LIBS})
target_compile_options(statistical_knowledge_graph PRIVATE ${UG_C_FLAGS})

# Example 3: Basic Usage (would be implemented)
# add_executable(basic_usage basic_usage.c)
# target_link_libraries(basic_usage ${UG_LIBRARY} ${UG_PLATFORM_LIBS})

# Example 4: Language Migration Demo (would be implemented)  
# add_executable(migration_demo migration_demo.c)
# target_link_libraries(migration_demo ${UG_LIBRARY} ${UG_PLATFORM_LIBS})

# Install examples
install(TARGETS infinite_complexity_demo statistical_knowledge_graph
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples
)

# Install source files and scripts for reference
install(FILES 
    infinite_complexity_demo.c
    statistical_knowledge_graph.c
    cli_demo.ug
    complex_graph_demo.ug
    tutorial_basic.ug
    social_network.ug
    DESTINATION ${CMAKE_INSTALL_DOCDIR}/examples
)