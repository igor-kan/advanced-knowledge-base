/**
 * Universal Graph Engine - UI Management and Visualization
 * 
 * This module handles the D3.js-based visualization, user interactions,
 * and UI state management for the graph interface.
 */

class GraphUI {
    constructor(engine, containerId) {
        this.engine = engine;
        this.containerId = containerId;
        this.container = d3.select(`#${containerId}`);
        
        // UI State
        this.currentTool = 'select';
        this.selectedNodes = new Set();
        this.selectedEdges = new Set();
        this.isDragging = false;
        this.isConnecting = false;
        this.connectionSource = null;
        this.hoveredNode = null;
        this.hoveredEdge = null;
        
        // Visualization settings
        this.settings = {
            width: 1200,
            height: 800,
            animationDuration: 300,
            animationSpeed: 1.0,
            layoutType: 'force',
            showGrid: true,
            showLabels: true,
            enablePhysics: true
        };
        
        // D3 components
        this.svg = null;
        this.simulation = null;
        this.zoom = null;
        this.transform = d3.zoomIdentity;
        
        // Initialize
        this.initializeVisualization();
        this.bindEvents();
        this.startSimulation();
    }

    initializeVisualization() {
        // Get container dimensions
        const rect = this.container.node().getBoundingClientRect();
        this.settings.width = rect.width;
        this.settings.height = rect.height;

        // Setup SVG
        this.svg = this.container.select('svg')
            .attr('width', this.settings.width)
            .attr('height', this.settings.height);

        // Setup zoom behavior
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on('zoom', (event) => {
                this.transform = event.transform;
                this.svg.select('#graph-container')
                    .attr('transform', this.transform);
            });

        this.svg.call(this.zoom);

        // Get groups for nodes and edges
        this.edgesGroup = this.svg.select('#edges-group');
        this.nodesGroup = this.svg.select('#nodes-group');

        // Handle canvas clicks
        this.svg.on('click', (event) => {
            if (event.target === this.svg.node() || event.target.id === 'graph-container') {
                this.handleCanvasClick(event);
            }
        });

        // Handle right-click context menu
        this.svg.on('contextmenu', (event) => {
            event.preventDefault();
            this.hideContextMenu();
        });
    }

    bindEvents() {
        // Listen to graph engine events
        this.engine.addEventListener('nodeAdded', (data) => {
            this.addNodeToVisualization(data.node);
            this.updateStatistics();
        });

        this.engine.addEventListener('nodeUpdated', (data) => {
            this.updateNodeVisualization(data.node);
        });

        this.engine.addEventListener('nodeDeleted', (data) => {
            this.removeNodeFromVisualization(data.nodeId);
            this.updateStatistics();
        });

        this.engine.addEventListener('edgeAdded', (data) => {
            this.addEdgeToVisualization(data.edge);
            this.updateStatistics();
        });

        this.engine.addEventListener('edgeDeleted', (data) => {
            this.removeEdgeFromVisualization(data.edgeId);
            this.updateStatistics();
        });

        this.engine.addEventListener('graphCleared', () => {
            this.clearVisualization();
            this.updateStatistics();
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });
    }

    startSimulation() {
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(100).strength(0.3))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(this.settings.width / 2, this.settings.height / 2))
            .force('collision', d3.forceCollide().radius(d => d.size / 2 + 5))
            .alphaDecay(0.02)
            .on('tick', () => {
                this.updatePositions();
            });

        this.updateSimulation();
    }

    updateSimulation() {
        if (!this.simulation) return;

        const nodes = this.engine.getAllNodes();
        const edges = this.engine.getAllEdges();

        // Update simulation data
        this.simulation.nodes(nodes);
        this.simulation.force('link').links(edges.map(e => ({
            ...e,
            source: e.source,
            target: e.target
        })));

        // Restart simulation
        this.simulation.alpha(0.3).restart();
    }

    // Node visualization methods
    addNodeToVisualization(node) {
        const nodeGroup = this.nodesGroup
            .append('g')
            .attr('class', 'node')
            .attr('id', `node-${node.id}`)
            .style('cursor', 'pointer')
            .call(this.createNodeDragBehavior());

        // Node circle
        nodeGroup
            .append('circle')
            .attr('r', node.size / 2)
            .attr('fill', node.color)
            .attr('stroke', '#ffffff')
            .attr('stroke-width', 2);

        // Node label
        if (this.settings.showLabels) {
            nodeGroup
                .append('text')
                .text(node.label)
                .attr('text-anchor', 'middle')
                .attr('dy', '.35em')
                .style('font-size', '12px')
                .style('font-weight', '500')
                .style('fill', '#1e293b')
                .style('pointer-events', 'none')
                .style('user-select', 'none');
        }

        // Event handlers
        nodeGroup
            .on('click', (event, d) => {
                event.stopPropagation();
                this.handleNodeClick(event, node);
            })
            .on('mouseenter', (event, d) => {
                this.handleNodeMouseEnter(event, node);
            })
            .on('mouseleave', (event, d) => {
                this.handleNodeMouseLeave(event, node);
            })
            .on('contextmenu', (event, d) => {
                event.preventDefault();
                event.stopPropagation();
                this.showContextMenu(event, node);
            });

        this.updateSimulation();
    }

    updateNodeVisualization(node) {
        const nodeGroup = this.svg.select(`#node-${node.id}`);
        
        nodeGroup.select('circle')
            .transition()
            .duration(this.settings.animationDuration)
            .attr('r', node.size / 2)
            .attr('fill', node.color);

        nodeGroup.select('text')
            .text(node.label);
    }

    removeNodeFromVisualization(nodeId) {
        this.svg.select(`#node-${nodeId}`)
            .transition()
            .duration(this.settings.animationDuration)
            .style('opacity', 0)
            .remove();

        this.selectedNodes.delete(nodeId);
        this.updateSimulation();
    }

    // Edge visualization methods
    addEdgeToVisualization(edge) {
        const edgeGroup = this.edgesGroup
            .append('g')
            .attr('class', 'edge')
            .attr('id', `edge-${edge.id}`)
            .style('cursor', 'pointer');

        // Edge line
        const line = edgeGroup
            .append('line')
            .attr('stroke', edge.color || '#64748b')
            .attr('stroke-width', 2)
            .attr('fill', 'none');

        // Add arrowhead for directed edges
        if (edge.directed) {
            line.attr('marker-end', 'url(#arrowhead)');
        }

        // Edge label
        if (edge.label) {
            edgeGroup
                .append('text')
                .attr('class', 'edge-label')
                .text(edge.label)
                .attr('text-anchor', 'middle')
                .attr('dy', '-5')
                .style('font-size', '10px')
                .style('font-weight', '500')
                .style('fill', '#64748b')
                .style('pointer-events', 'none')
                .style('user-select', 'none');
        }

        // Event handlers
        edgeGroup
            .on('click', (event, d) => {
                event.stopPropagation();
                this.handleEdgeClick(event, edge);
            })
            .on('mouseenter', (event, d) => {
                this.handleEdgeMouseEnter(event, edge);
            })
            .on('mouseleave', (event, d) => {
                this.handleEdgeMouseLeave(event, edge);
            });

        this.updateSimulation();
    }

    removeEdgeFromVisualization(edgeId) {
        this.svg.select(`#edge-${edgeId}`)
            .transition()
            .duration(this.settings.animationDuration)
            .style('opacity', 0)
            .remove();

        this.selectedEdges.delete(edgeId);
        this.updateSimulation();
    }

    updatePositions() {
        // Update node positions
        this.nodesGroup.selectAll('.node')
            .attr('transform', d => {
                const node = this.engine.getNode(d.id) || d;
                return `translate(${node.x}, ${node.y})`;
            });

        // Update edge positions
        this.edgesGroup.selectAll('.edge').each((d, i, nodes) => {
            const edgeElement = d3.select(nodes[i]);
            const edge = this.engine.getEdge(d.id) || d;
            
            const sourceNode = this.engine.getNode(edge.source) || { x: 0, y: 0 };
            const targetNode = this.engine.getNode(edge.target) || { x: 0, y: 0 };

            // Update line position
            edgeElement.select('line')
                .attr('x1', sourceNode.x)
                .attr('y1', sourceNode.y)
                .attr('x2', targetNode.x)
                .attr('y2', targetNode.y);

            // Update label position
            const labelElement = edgeElement.select('.edge-label');
            if (!labelElement.empty()) {
                const midX = (sourceNode.x + targetNode.x) / 2;
                const midY = (sourceNode.y + targetNode.y) / 2;
                labelElement
                    .attr('x', midX)
                    .attr('y', midY);
            }
        });
    }

    // Drag behavior
    createNodeDragBehavior() {
        return d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                const node = this.engine.getNode(d.id) || d;
                node.fx = node.x;
                node.fy = node.y;
                this.isDragging = true;
            })
            .on('drag', (event, d) => {
                const node = this.engine.getNode(d.id) || d;
                node.fx = event.x;
                node.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                const node = this.engine.getNode(d.id) || d;
                if (!event.sourceEvent.shiftKey) {
                    node.fx = null;
                    node.fy = null;
                }
                this.isDragging = false;
            });
    }

    // Event handlers
    handleCanvasClick(event) {
        if (this.isDragging) return;

        // Get click position relative to the SVG
        const [x, y] = d3.pointer(event, this.svg.select('#graph-container').node());

        switch (this.currentTool) {
            case 'node':
                this.createNodeAtPosition(x, y);
                break;
            case 'select':
                this.clearSelection();
                break;
        }
    }

    handleNodeClick(event, node) {
        if (this.isDragging) return;

        switch (this.currentTool) {
            case 'select':
                this.toggleNodeSelection(node);
                break;
            case 'edge':
                this.handleEdgeToolNodeClick(node);
                break;
            default:
                this.selectNode(node);
        }
    }

    handleEdgeClick(event, edge) {
        if (this.isDragging) return;

        switch (this.currentTool) {
            case 'select':
                this.toggleEdgeSelection(edge);
                break;
            case 'insert':
                this.insertNodeBetweenEdge(edge);
                break;
            case 'triangle':
                this.createTriangleFromEdge(edge);
                break;
            default:
                this.selectEdge(edge);
        }
    }

    handleNodeMouseEnter(event, node) {
        this.hoveredNode = node;
        
        // Add hover effect
        const nodeElement = this.svg.select(`#node-${node.id} circle`);
        nodeElement
            .transition()
            .duration(150)
            .attr('stroke-width', 3)
            .style('filter', 'drop-shadow(0 4px 8px rgba(0, 0, 0, 0.15))');
    }

    handleNodeMouseLeave(event, node) {
        this.hoveredNode = null;
        
        // Remove hover effect if not selected
        if (!this.selectedNodes.has(node.id)) {
            const nodeElement = this.svg.select(`#node-${node.id} circle`);
            nodeElement
                .transition()
                .duration(150)
                .attr('stroke-width', 2)
                .style('filter', 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1))');
        }
    }

    handleEdgeMouseEnter(event, edge) {
        this.hoveredEdge = edge;
        
        // Add hover effect
        const edgeElement = this.svg.select(`#edge-${edge.id} line`);
        edgeElement
            .transition()
            .duration(150)
            .attr('stroke', '#3b82f6')
            .attr('stroke-width', 3);
    }

    handleEdgeMouseLeave(event, edge) {
        this.hoveredEdge = null;
        
        // Remove hover effect if not selected
        if (!this.selectedEdges.has(edge.id)) {
            const edgeElement = this.svg.select(`#edge-${edge.id} line`);
            edgeElement
                .transition()
                .duration(150)
                .attr('stroke', edge.color || '#64748b')
                .attr('stroke-width', 2);
        }
    }

    // Tool-specific handlers
    handleEdgeToolNodeClick(node) {
        if (!this.isConnecting) {
            // Start connection
            this.isConnecting = true;
            this.connectionSource = node;
            this.selectNode(node);
            this.showToast('Select target node to create connection', 'info');
        } else {
            // Complete connection
            if (node.id !== this.connectionSource.id) {
                this.createEdgeBetweenNodes(this.connectionSource, node);
            }
            this.isConnecting = false;
            this.connectionSource = null;
            this.clearSelection();
        }
    }

    // Graph operations
    createNodeAtPosition(x, y) {
        const node = this.engine.createNode({
            x: x,
            y: y,
            label: `Node ${this.engine.nodeIdCounter}`,
            fx: null,
            fy: null
        });
        
        this.showToast(`Created node: ${node.label}`, 'success');
    }

    createEdgeBetweenNodes(sourceNode, targetNode) {
        try {
            const edge = this.engine.createEdge(sourceNode.id, targetNode.id, {
                label: 'CONNECTED'
            });
            this.showToast(`Connected ${sourceNode.label} to ${targetNode.label}`, 'success');
        } catch (error) {
            this.showToast(error.message, 'error');
        }
    }

    insertNodeBetweenEdge(edge) {
        try {
            const result = this.engine.insertNodeBetween(edge.id);
            this.showToast(`Inserted node between connected nodes`, 'success');
        } catch (error) {
            this.showToast(error.message, 'error');
        }
    }

    createTriangleFromEdge(edge) {
        try {
            const result = this.engine.createTriangle(edge.id);
            this.showToast(`Created triangular connection`, 'success');
        } catch (error) {
            this.showToast(error.message, 'error');
        }
    }

    // Selection management
    selectNode(node) {
        this.clearSelection();
        this.selectedNodes.add(node.id);
        this.updateNodeSelection(node.id, true);
        this.updatePropertiesPanel(node);
    }

    toggleNodeSelection(node) {
        if (this.selectedNodes.has(node.id)) {
            this.selectedNodes.delete(node.id);
            this.updateNodeSelection(node.id, false);
        } else {
            this.selectedNodes.add(node.id);
            this.updateNodeSelection(node.id, true);
            this.updatePropertiesPanel(node);
        }
    }

    selectEdge(edge) {
        this.clearSelection();
        this.selectedEdges.add(edge.id);
        this.updateEdgeSelection(edge.id, true);
    }

    toggleEdgeSelection(edge) {
        if (this.selectedEdges.has(edge.id)) {
            this.selectedEdges.delete(edge.id);
            this.updateEdgeSelection(edge.id, false);
        } else {
            this.selectedEdges.add(edge.id);
            this.updateEdgeSelection(edge.id, true);
        }
    }

    clearSelection() {
        // Clear node selection
        this.selectedNodes.forEach(nodeId => {
            this.updateNodeSelection(nodeId, false);
        });
        this.selectedNodes.clear();

        // Clear edge selection
        this.selectedEdges.forEach(edgeId => {
            this.updateEdgeSelection(edgeId, false);
        });
        this.selectedEdges.clear();

        // Clear properties panel
        this.clearPropertiesPanel();
    }

    updateNodeSelection(nodeId, selected) {
        const nodeElement = this.svg.select(`#node-${nodeId} circle`);
        
        if (selected) {
            nodeElement
                .classed('selected', true)
                .attr('stroke', '#3b82f6')
                .attr('stroke-width', 4)
                .style('filter', 'url(#glow)');
        } else {
            nodeElement
                .classed('selected', false)
                .attr('stroke', '#ffffff')
                .attr('stroke-width', 2)
                .style('filter', 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1))');
        }
    }

    updateEdgeSelection(edgeId, selected) {
        const edgeElement = this.svg.select(`#edge-${edgeId} line`);
        
        if (selected) {
            edgeElement
                .classed('selected', true)
                .attr('stroke', '#3b82f6')
                .attr('stroke-width', 4);
        } else {
            const edge = this.engine.getEdge(edgeId);
            edgeElement
                .classed('selected', false)
                .attr('stroke', edge?.color || '#64748b')
                .attr('stroke-width', 2);
        }
    }

    // UI management
    setCurrentTool(tool) {
        this.currentTool = tool;
        
        // Reset connection state when changing tools
        if (tool !== 'edge') {
            this.isConnecting = false;
            this.connectionSource = null;
        }
        
        // Update cursor
        this.updateCursor();
        
        // Show tool instructions
        this.showToolInstructions(tool);
    }

    updateCursor() {
        let cursor = 'default';
        
        switch (this.currentTool) {
            case 'select':
                cursor = 'default';
                break;
            case 'node':
                cursor = 'crosshair';
                break;
            case 'edge':
                cursor = this.isConnecting ? 'crosshair' : 'default';
                break;
            case 'insert':
            case 'triangle':
                cursor = 'pointer';
                break;
        }
        
        this.svg.style('cursor', cursor);
    }

    showToolInstructions(tool) {
        const instructionsElement = document.getElementById('tool-instructions');
        const instructions = instructionsElement.querySelectorAll('.instruction');
        
        instructions.forEach(instruction => {
            instruction.classList.remove('active');
        });
        
        const activeInstruction = instructionsElement.querySelector(`[data-tool="${tool}"]`);
        if (activeInstruction) {
            activeInstruction.classList.add('active');
            instructionsElement.classList.add('show');
            
            // Hide after 3 seconds
            setTimeout(() => {
                instructionsElement.classList.remove('show');
            }, 3000);
        }
    }

    // Context menu
    showContextMenu(event, node) {
        const contextMenu = document.getElementById('context-menu');
        const rect = this.container.node().getBoundingClientRect();
        
        contextMenu.classList.add('show');
        contextMenu.style.left = `${event.clientX - rect.left}px`;
        contextMenu.style.top = `${event.clientY - rect.top}px`;
        
        // Store reference to the node
        contextMenu.dataset.nodeId = node.id;
    }

    hideContextMenu() {
        const contextMenu = document.getElementById('context-menu');
        contextMenu.classList.remove('show');
        delete contextMenu.dataset.nodeId;
    }

    // Properties panel
    updatePropertiesPanel(node) {
        const propertiesPanel = document.getElementById('node-properties');
        
        propertiesPanel.innerHTML = `
            <div class="property-item">
                <label>Label</label>
                <input type="text" id="prop-label" value="${node.label}">
            </div>
            <div class="property-item">
                <label>Description</label>
                <textarea id="prop-description" rows="3">${node.description || ''}</textarea>
            </div>
            <div class="property-item">
                <label>Color</label>
                <input type="color" id="prop-color" value="${node.color}">
            </div>
            <div class="property-item">
                <label>Size</label>
                <input type="range" id="prop-size" min="20" max="80" value="${node.size}">
                <span>${node.size}px</span>
            </div>
        `;

        // Bind change events
        const labelInput = document.getElementById('prop-label');
        const descInput = document.getElementById('prop-description');
        const colorInput = document.getElementById('prop-color');
        const sizeInput = document.getElementById('prop-size');

        const updateNode = () => {
            this.engine.updateNode(node.id, {
                label: labelInput.value,
                description: descInput.value,
                color: colorInput.value,
                size: parseInt(sizeInput.value)
            });
        };

        labelInput.addEventListener('input', updateNode);
        descInput.addEventListener('input', updateNode);
        colorInput.addEventListener('change', updateNode);
        sizeInput.addEventListener('input', (e) => {
            e.target.nextElementSibling.textContent = `${e.target.value}px`;
            updateNode();
        });
    }

    clearPropertiesPanel() {
        const propertiesPanel = document.getElementById('node-properties');
        propertiesPanel.innerHTML = `
            <div class="no-selection">
                <i class="fas fa-mouse-pointer"></i>
                <p>Click a node to edit its properties</p>
            </div>
        `;
    }

    // Statistics
    updateStatistics() {
        const stats = this.engine.getStatistics();
        
        document.getElementById('node-count').textContent = stats.nodeCount;
        document.getElementById('edge-count').textContent = stats.edgeCount;
        document.getElementById('density').textContent = stats.density.toFixed(4);
    }

    // Visualization controls
    fitToView() {
        const nodes = this.engine.getAllNodes();
        if (nodes.length === 0) return;

        const bounds = this.calculateBounds(nodes);
        const fullWidth = this.settings.width;
        const fullHeight = this.settings.height;
        const width = bounds.maxX - bounds.minX;
        const height = bounds.maxY - bounds.minY;

        if (width === 0 || height === 0) return;

        const midX = (bounds.minX + bounds.maxX) / 2;
        const midY = (bounds.minY + bounds.maxY) / 2;
        const scale = 0.8 / Math.max(width / fullWidth, height / fullHeight);
        const translate = [fullWidth / 2 - scale * midX, fullHeight / 2 - scale * midY];

        this.svg.transition()
            .duration(750)
            .call(this.zoom.transform, d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale));
    }

    calculateBounds(nodes) {
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        
        nodes.forEach(node => {
            minX = Math.min(minX, node.x - node.size / 2);
            minY = Math.min(minY, node.y - node.size / 2);
            maxX = Math.max(maxX, node.x + node.size / 2);
            maxY = Math.max(maxY, node.y + node.size / 2);
        });
        
        return { minX, minY, maxX, maxY };
    }

    zoomIn() {
        this.svg.transition().duration(300).call(this.zoom.scaleBy, 1.5);
    }

    zoomOut() {
        this.svg.transition().duration(300).call(this.zoom.scaleBy, 1 / 1.5);
    }

    // Utility methods
    clearVisualization() {
        this.nodesGroup.selectAll('.node').remove();
        this.edgesGroup.selectAll('.edge').remove();
        this.clearSelection();
    }

    handleResize() {
        const rect = this.container.node().getBoundingClientRect();
        this.settings.width = rect.width;
        this.settings.height = rect.height;

        this.svg
            .attr('width', this.settings.width)
            .attr('height', this.settings.height);

        if (this.simulation) {
            this.simulation
                .force('center', d3.forceCenter(this.settings.width / 2, this.settings.height / 2))
                .restart();
        }
    }

    showToast(message, type = 'info') {
        // This will be implemented by the main app
        if (window.showToast) {
            window.showToast(message, type);
        } else {
            console.log(`${type.toUpperCase()}: ${message}`);
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GraphUI;
} else if (typeof window !== 'undefined') {
    window.GraphUI = GraphUI;
}