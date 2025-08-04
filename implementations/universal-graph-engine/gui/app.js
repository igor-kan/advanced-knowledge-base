/**
 * Universal Graph Engine - Main Application Controller
 * 
 * This is the main application controller that ties together the graph engine,
 * UI components, and user interactions.
 */

class UniversalGraphApp {
    constructor() {
        this.engine = new UniversalGraphEngine();
        this.ui = null;
        this.currentModal = null;
        this.currentNodeForEdit = null;
        this.connectionTarget = null;
        
        // Initialize the application
        this.init();
    }

    init() {
        // Initialize UI
        this.ui = new GraphUI(this.engine, 'graph-canvas');
        
        // Bind UI events
        this.bindToolbarEvents();
        this.bindModalEvents();
        this.bindControlEvents();
        this.bindContextMenuEvents();
        this.bindKeyboardEvents();
        
        // Load sample data for demonstration
        this.loadSampleData();
        
        // Update initial statistics
        this.ui.updateStatistics();
        
        console.log('Universal Graph Engine GUI initialized');
    }

    bindToolbarEvents() {
        // Tool selection
        document.querySelectorAll('.tool-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tool = btn.dataset.tool;
                if (tool) {
                    this.setActiveTool(tool);
                }
            });
        });

        // Zoom controls
        document.getElementById('zoom-in').addEventListener('click', () => {
            this.ui.zoomIn();
        });

        document.getElementById('zoom-out').addEventListener('click', () => {
            this.ui.zoomOut();
        });

        document.getElementById('fit-view').addEventListener('click', () => {
            this.ui.fitToView();
        });

        // Header buttons
        document.getElementById('save-btn').addEventListener('click', () => {
            this.saveGraph();
        });

        document.getElementById('export-btn').addEventListener('click', () => {
            this.exportGraph();
        });
    }

    bindModalEvents() {
        // Node edit modal
        const nodeModal = document.getElementById('node-modal');
        const nodeModalClose = nodeModal.querySelector('.modal-close');
        const cancelEdit = document.getElementById('cancel-edit');
        const saveNode = document.getElementById('save-node');

        nodeModalClose.addEventListener('click', () => this.hideModal('node-modal'));
        cancelEdit.addEventListener('click', () => this.hideModal('node-modal'));
        saveNode.addEventListener('click', () => this.saveNodeEdits());

        // Connection modal
        const connectionModal = document.getElementById('connection-modal');
        const connectionModalClose = connectionModal.querySelector('.modal-close');
        const cancelConnection = document.getElementById('cancel-connection');
        const createConnection = document.getElementById('create-connection');

        connectionModalClose.addEventListener('click', () => this.hideModal('connection-modal'));
        cancelConnection.addEventListener('click', () => this.hideModal('connection-modal'));
        createConnection.addEventListener('click', () => this.createConnection());

        // Color presets
        document.querySelectorAll('.color-preset').forEach(preset => {
            preset.addEventListener('click', (e) => {
                const color = e.target.dataset.color;
                document.getElementById('node-color').value = color;
            });
        });

        // Size slider
        document.getElementById('node-size').addEventListener('input', (e) => {
            document.getElementById('size-value').textContent = `${e.target.value}px`;
        });

        // Animation speed slider
        document.getElementById('animation-speed').addEventListener('input', (e) => {
            const speed = parseFloat(e.target.value);
            document.getElementById('speed-value').textContent = `${speed}x`;
            this.ui.settings.animationSpeed = speed;
        });

        // Close modals when clicking outside
        document.querySelectorAll('.modal').forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.hideModal(modal.id);
                }
            });
        });
    }

    bindControlEvents() {
        // Layout selection
        document.getElementById('layout-select').addEventListener('change', (e) => {
            this.changeLayout(e.target.value);
        });

        // Clear graph
        document.getElementById('clear-graph').addEventListener('click', () => {
            this.clearGraph();
        });
    }

    bindContextMenuEvents() {
        const contextMenu = document.getElementById('context-menu');
        
        // Hide context menu when clicking elsewhere
        document.addEventListener('click', () => {
            this.ui.hideContextMenu();
        });

        // Context menu actions
        contextMenu.addEventListener('click', (e) => {
            const action = e.target.closest('.menu-item')?.dataset.action;
            const nodeId = contextMenu.dataset.nodeId;
            
            if (action && nodeId) {
                this.handleContextMenuAction(action, nodeId);
            }
            
            this.ui.hideContextMenu();
        });
    }

    bindKeyboardEvents() {
        document.addEventListener('keydown', (e) => {
            // Don't handle shortcuts when typing in inputs
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }

            switch (e.key.toLowerCase()) {
                case 's':
                    if (!e.ctrlKey && !e.metaKey) {
                        e.preventDefault();
                        this.setActiveTool('select');
                    }
                    break;
                case 'n':
                    e.preventDefault();
                    this.setActiveTool('node');
                    break;
                case 'e':
                    e.preventDefault();
                    this.setActiveTool('edge');
                    break;
                case 'i':
                    e.preventDefault();
                    this.setActiveTool('insert');
                    break;
                case 't':
                    e.preventDefault();
                    this.setActiveTool('triangle');
                    break;
                case 'f':
                    e.preventDefault();
                    this.ui.fitToView();
                    break;
                case '+':
                case '=':
                    e.preventDefault();
                    this.ui.zoomIn();
                    break;
                case '-':
                    e.preventDefault();
                    this.ui.zoomOut();
                    break;
                case 'delete':
                case 'backspace':
                    e.preventDefault();
                    this.deleteSelected();
                    break;
                case 'escape':
                    e.preventDefault();
                    this.ui.clearSelection();
                    if (this.currentModal) {
                        this.hideModal(this.currentModal);
                    }
                    break;
            }
        });
    }

    // Tool management
    setActiveTool(tool) {
        // Update UI
        document.querySelectorAll('.tool-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tool="${tool}"]`)?.classList.add('active');
        
        // Update UI component
        this.ui.setCurrentTool(tool);
        
        this.showToast(`Switched to ${tool} tool`, 'info');
    }

    // Context menu actions
    handleContextMenuAction(action, nodeId) {
        const node = this.engine.getNode(nodeId);
        if (!node) return;

        switch (action) {
            case 'edit':
                this.editNode(node);
                break;
            case 'connect':
                this.showConnectionModal(node);
                break;
            case 'insert':
                this.showToast('Click an edge to insert a node between connected nodes', 'info');
                this.setActiveTool('insert');
                break;
            case 'triangle':
                this.showToast('Click an edge to create a triangular connection', 'info');
                this.setActiveTool('triangle');
                break;
            case 'delete':
                this.deleteNode(node);
                break;
        }
    }

    // Node operations
    editNode(node) {
        this.currentNodeForEdit = node;
        
        // Populate modal fields
        document.getElementById('node-label').value = node.label;
        document.getElementById('node-description').value = node.description || '';
        document.getElementById('node-color').value = node.color;
        document.getElementById('node-size').value = node.size;
        document.getElementById('size-value').textContent = `${node.size}px`;
        
        this.showModal('node-modal');
    }

    saveNodeEdits() {
        if (!this.currentNodeForEdit) return;

        const updates = {
            label: document.getElementById('node-label').value,
            description: document.getElementById('node-description').value,
            color: document.getElementById('node-color').value,
            size: parseInt(document.getElementById('node-size').value)
        };

        try {
            this.engine.updateNode(this.currentNodeForEdit.id, updates);
            this.hideModal('node-modal');
            this.showToast('Node updated successfully', 'success');
        } catch (error) {
            this.showToast(error.message, 'error');
        }
    }

    deleteNode(node) {
        if (confirm(`Are you sure you want to delete "${node.label}"?`)) {
            try {
                this.engine.deleteNode(node.id);
                this.showToast(`Deleted node: ${node.label}`, 'success');
            } catch (error) {
                this.showToast(error.message, 'error');
            }
        }
    }

    deleteSelected() {
        const selectedNodes = Array.from(this.ui.selectedNodes);
        const selectedEdges = Array.from(this.ui.selectedEdges);
        
        if (selectedNodes.length === 0 && selectedEdges.length === 0) {
            this.showToast('No nodes or edges selected', 'warning');
            return;
        }

        const nodeText = selectedNodes.length === 1 ? '1 node' : `${selectedNodes.length} nodes`;
        const edgeText = selectedEdges.length === 1 ? '1 edge' : `${selectedEdges.length} edges`;
        let confirmText = 'Are you sure you want to delete ';

        if (selectedNodes.length > 0 && selectedEdges.length > 0) {
            confirmText += `${nodeText} and ${edgeText}?`;
        } else if (selectedNodes.length > 0) {
            confirmText += `${nodeText}?`;
        } else {
            confirmText += `${edgeText}?`;
        }

        if (confirm(confirmText)) {
            try {
                selectedEdges.forEach(edgeId => {
                    this.engine.deleteEdge(edgeId);
                });
                
                selectedNodes.forEach(nodeId => {
                    this.engine.deleteNode(nodeId);
                });
                
                this.showToast('Selected items deleted', 'success');
            } catch (error) {
                this.showToast(error.message, 'error');
            }
        }
    }

    // Connection modal
    showConnectionModal(sourceNode) {
        this.connectionSource = sourceNode;
        
        // Populate available nodes
        const availableNodes = document.getElementById('available-nodes');
        const allNodes = this.engine.getAllNodes().filter(n => n.id !== sourceNode.id);
        
        availableNodes.innerHTML = allNodes.map(node => `
            <div class="node-list-item" data-node-id="${node.id}">
                <div class="node-preview" style="background: ${node.color}"></div>
                <span>${node.label}</span>
            </div>
        `).join('');

        // Bind selection events
        availableNodes.querySelectorAll('.node-list-item').forEach(item => {
            item.addEventListener('click', () => {
                availableNodes.querySelectorAll('.node-list-item').forEach(i => 
                    i.classList.remove('selected'));
                item.classList.add('selected');
                this.connectionTarget = item.dataset.nodeId;
            });
        });

        this.showModal('connection-modal');
    }

    createConnection() {
        if (!this.connectionSource || !this.connectionTarget) {
            this.showToast('Please select a target node', 'warning');
            return;
        }

        const edgeLabel = document.getElementById('edge-label').value || 'CONNECTED';
        const edgeWeight = parseFloat(document.getElementById('edge-weight').value) || 1.0;

        try {
            const edge = this.engine.createEdge(this.connectionSource.id, this.connectionTarget, {
                label: edgeLabel,
                weight: edgeWeight
            });
            
            this.hideModal('connection-modal');
            this.showToast('Connection created successfully', 'success');
        } catch (error) {
            this.showToast(error.message, 'error');
        }
    }

    // Layout management
    changeLayout(layoutType) {
        this.ui.settings.layoutType = layoutType;
        
        switch (layoutType) {
            case 'force':
                this.applyForceLayout();
                break;
            case 'circular':
                this.applyCircularLayout();
                break;
            case 'hierarchical':
                this.applyHierarchicalLayout();
                break;
            case 'grid':
                this.applyGridLayout();
                break;
        }
        
        this.showToast(`Applied ${layoutType} layout`, 'success');
    }

    applyForceLayout() {
        // Reset simulation with default forces
        if (this.ui.simulation) {
            this.ui.simulation
                .force('link', d3.forceLink().id(d => d.id).distance(100).strength(0.3))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(this.ui.settings.width / 2, this.ui.settings.height / 2))
                .force('collision', d3.forceCollide().radius(d => d.size / 2 + 5))
                .alpha(0.3)
                .restart();
        }
    }

    applyCircularLayout() {
        const nodes = this.engine.getAllNodes();
        const centerX = this.ui.settings.width / 2;
        const centerY = this.ui.settings.height / 2;
        const radius = Math.min(this.ui.settings.width, this.ui.settings.height) / 3;
        
        nodes.forEach((node, index) => {
            const angle = (2 * Math.PI * index) / nodes.length;
            this.engine.updateNode(node.id, {
                x: centerX + radius * Math.cos(angle),
                y: centerY + radius * Math.sin(angle),
                fx: centerX + radius * Math.cos(angle),
                fy: centerY + radius * Math.sin(angle)
            });
        });

        // Disable physics temporarily
        if (this.ui.simulation) {
            this.ui.simulation.alpha(0.1).restart();
        }
    }

    applyHierarchicalLayout() {
        const nodes = this.engine.getAllNodes();
        const components = this.engine.findConnectedComponents();
        
        let currentY = 50;
        const levelHeight = 100;
        
        components.forEach(component => {
            const componentNodes = component.map(id => this.engine.getNode(id));
            const levels = this.calculateHierarchicalLevels(componentNodes);
            
            levels.forEach((levelNodes, levelIndex) => {
                const levelY = currentY + levelIndex * levelHeight;
                const startX = (this.ui.settings.width - (levelNodes.length - 1) * 150) / 2;
                
                levelNodes.forEach((node, nodeIndex) => {
                    this.engine.updateNode(node.id, {
                        x: startX + nodeIndex * 150,
                        y: levelY,
                        fx: startX + nodeIndex * 150,
                        fy: levelY
                    });
                });
            });
            
            currentY += levels.length * levelHeight + 50;
        });

        if (this.ui.simulation) {
            this.ui.simulation.alpha(0.1).restart();
        }
    }

    applyGridLayout() {
        const nodes = this.engine.getAllNodes();
        const cols = Math.ceil(Math.sqrt(nodes.length));
        const cellWidth = this.ui.settings.width / cols;
        const cellHeight = this.ui.settings.height / Math.ceil(nodes.length / cols);
        
        nodes.forEach((node, index) => {
            const col = index % cols;
            const row = Math.floor(index / cols);
            
            this.engine.updateNode(node.id, {
                x: col * cellWidth + cellWidth / 2,
                y: row * cellHeight + cellHeight / 2,
                fx: col * cellWidth + cellWidth / 2,
                fy: row * cellHeight + cellHeight / 2
            });
        });

        if (this.ui.simulation) {
            this.ui.simulation.alpha(0.1).restart();
        }
    }

    calculateHierarchicalLevels(nodes) {
        // Simple level calculation based on node degree
        const levels = [];
        const processed = new Set();
        
        // Find root nodes (nodes with no incoming edges)
        const rootNodes = nodes.filter(node => {
            const edges = this.engine.getNodeEdges(node.id);
            const incomingEdges = edges.filter(edge => edge.target === node.id);
            return incomingEdges.length === 0;
        });
        
        if (rootNodes.length === 0 && nodes.length > 0) {
            rootNodes.push(nodes[0]); // If no clear roots, start with first node
        }
        
        let currentLevel = rootNodes;
        
        while (currentLevel.length > 0) {
            levels.push([...currentLevel]);
            currentLevel.forEach(node => processed.add(node.id));
            
            const nextLevel = [];
            currentLevel.forEach(node => {
                const neighbors = this.engine.getNodeNeighbors(node.id);
                neighbors.forEach(neighbor => {
                    if (!processed.has(neighbor.id) && !nextLevel.find(n => n.id === neighbor.id)) {
                        nextLevel.push(neighbor);
                    }
                });
            });
            
            currentLevel = nextLevel;
        }
        
        return levels;
    }

    // Graph operations
    clearGraph() {
        if (confirm('Are you sure you want to clear the entire graph?')) {
            this.engine.clear();
            this.showToast('Graph cleared', 'success');
        }
    }

    saveGraph() {
        const graphData = this.engine.exportToJSON();
        const dataStr = JSON.stringify(graphData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `graph_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
        link.click();
        
        this.showToast('Graph saved successfully', 'success');
    }

    exportGraph() {
        const graphData = this.engine.exportToJSON();
        
        // Create DOT format for Graphviz
        let dotContent = 'digraph UniversalGraph {\n';
        dotContent += '  rankdir=LR;\n';
        dotContent += '  node [shape=circle, style=filled];\n\n';
        
        // Add nodes
        graphData.nodes.forEach(node => {
            dotContent += `  "${node.id}" [label="${node.label}", fillcolor="${node.color}"];\n`;
        });
        
        dotContent += '\n';
        
        // Add edges
        graphData.edges.forEach(edge => {
            const arrow = edge.directed ? '->' : '--';
            const label = edge.label ? ` [label="${edge.label}"]` : '';
            dotContent += `  "${edge.source}" ${arrow} "${edge.target}"${label};\n`;
        });
        
        dotContent += '}';
        
        // Download DOT file
        const dotBlob = new Blob([dotContent], { type: 'text/plain' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dotBlob);
        link.download = `graph_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.dot`;
        link.click();
        
        this.showToast('Graph exported as DOT file', 'success');
    }

    // Modal management
    showModal(modalId) {
        const modal = document.getElementById(modalId);
        modal.classList.add('show');
        this.currentModal = modalId;
        
        // Focus first input
        const firstInput = modal.querySelector('input, textarea');
        if (firstInput) {
            setTimeout(() => firstInput.focus(), 100);
        }
    }

    hideModal(modalId) {
        const modal = document.getElementById(modalId);
        modal.classList.remove('show');
        
        if (this.currentModal === modalId) {
            this.currentModal = null;
        }
        
        // Reset form if needed
        const form = modal.querySelector('form');
        if (form) {
            form.reset();
        }
    }

    // Sample data
    loadSampleData() {
        // Create some sample nodes
        const alice = this.engine.createNode({
            label: 'Alice',
            description: 'Software Engineer',
            color: '#3b82f6',
            size: 50,
            x: 300,
            y: 200
        });

        const bob = this.engine.createNode({
            label: 'Bob',
            description: 'Data Scientist',
            color: '#10b981',
            size: 45,
            x: 500,
            y: 200
        });

        const charlie = this.engine.createNode({
            label: 'Charlie',
            description: 'Product Manager',
            color: '#f59e0b',
            size: 40,
            x: 400,
            y: 350
        });

        const diana = this.engine.createNode({
            label: 'Diana',
            description: 'UX Designer',
            color: '#ec4899',
            size: 45,
            x: 600,
            y: 350
        });

        // Create some connections
        this.engine.createEdge(alice.id, bob.id, {
            label: 'COLLABORATES',
            weight: 0.8
        });

        this.engine.createEdge(bob.id, charlie.id, {
            label: 'REPORTS_TO',
            weight: 0.9
        });

        this.engine.createEdge(charlie.id, diana.id, {
            label: 'WORKS_WITH',
            weight: 0.7
        });

        this.engine.createEdge(diana.id, alice.id, {
            label: 'FRIENDS',
            weight: 0.6
        });
    }

    // Toast notifications
    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icon = type === 'success' ? 'fas fa-check-circle' :
                    type === 'error' ? 'fas fa-exclamation-circle' :
                    type === 'warning' ? 'fas fa-exclamation-triangle' :
                    'fas fa-info-circle';
        
        toast.innerHTML = `
            <div class="toast-icon">
                <i class="${icon}"></i>
            </div>
            <div class="toast-content">
                <div class="toast-message">${message}</div>
            </div>
        `;
        
        toastContainer.appendChild(toast);
        
        // Show toast
        setTimeout(() => {
            toast.classList.add('show');
        }, 100);
        
        // Hide and remove toast
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, 3000);
    }
}

// Global toast function for UI components
window.showToast = function(message, type) {
    if (window.app) {
        window.app.showToast(message, type);
    }
};

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new UniversalGraphApp();
});