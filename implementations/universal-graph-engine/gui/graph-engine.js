/**
 * Universal Graph Engine - JavaScript Graph Data Management
 * 
 * This module provides the core graph data structure and operations
 * for the web-based GUI interface.
 */

class UniversalGraphEngine {
    constructor() {
        this.nodes = new Map();
        this.edges = new Map();
        this.nodeIdCounter = 0;
        this.edgeIdCounter = 0;
        this.eventListeners = new Map();
        
        // Graph properties
        this.properties = {
            directed: true,
            allowSelfLoops: true,
            allowMultipleEdges: false
        };
    }

    // Event system
    addEventListener(event, callback) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, []);
        }
        this.eventListeners.get(event).push(callback);
    }

    removeEventListener(event, callback) {
        if (this.eventListeners.has(event)) {
            const callbacks = this.eventListeners.get(event);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
    }

    emit(event, data) {
        if (this.eventListeners.has(event)) {
            this.eventListeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error('Error in event callback:', error);
                }
            });
        }
    }

    // Node operations
    createNode(data = {}) {
        const nodeId = `node_${++this.nodeIdCounter}`;
        const node = {
            id: nodeId,
            label: data.label || `Node ${this.nodeIdCounter}`,
            description: data.description || '',
            color: data.color || '#3b82f6',
            size: data.size || 40,
            x: data.x || Math.random() * 800 + 100,
            y: data.y || Math.random() * 600 + 100,
            vx: 0,
            vy: 0,
            fx: null, // Fixed position x
            fy: null, // Fixed position y
            properties: data.properties || {},
            created: new Date(),
            modified: new Date()
        };

        this.nodes.set(nodeId, node);
        this.emit('nodeAdded', { node });
        this.emit('graphChanged', { type: 'nodeAdded', node });
        
        return node;
    }

    updateNode(nodeId, updates) {
        const node = this.nodes.get(nodeId);
        if (!node) {
            throw new Error(`Node ${nodeId} not found`);
        }

        // Update node properties
        Object.assign(node, updates);
        node.modified = new Date();

        this.emit('nodeUpdated', { node, updates });
        this.emit('graphChanged', { type: 'nodeUpdated', node, updates });
        
        return node;
    }

    deleteNode(nodeId) {
        const node = this.nodes.get(nodeId);
        if (!node) {
            throw new Error(`Node ${nodeId} not found`);
        }

        // Remove all edges connected to this node
        const connectedEdges = this.getNodeEdges(nodeId);
        connectedEdges.forEach(edge => {
            this.deleteEdge(edge.id);
        });

        // Remove the node
        this.nodes.delete(nodeId);
        
        this.emit('nodeDeleted', { nodeId, node });
        this.emit('graphChanged', { type: 'nodeDeleted', nodeId, node });
        
        return node;
    }

    getNode(nodeId) {
        return this.nodes.get(nodeId);
    }

    getAllNodes() {
        return Array.from(this.nodes.values());
    }

    // Edge operations
    createEdge(fromNodeId, toNodeId, data = {}) {
        const fromNode = this.nodes.get(fromNodeId);
        const toNode = this.nodes.get(toNodeId);
        
        if (!fromNode) {
            throw new Error(`Source node ${fromNodeId} not found`);
        }
        if (!toNode) {
            throw new Error(`Target node ${toNodeId} not found`);
        }

        // Check for existing edge if multiple edges not allowed
        if (!this.properties.allowMultipleEdges) {
            const existingEdge = this.findEdge(fromNodeId, toNodeId);
            if (existingEdge) {
                throw new Error(`Edge already exists between ${fromNodeId} and ${toNodeId}`);
            }
        }

        const edgeId = `edge_${++this.edgeIdCounter}`;
        const edge = {
            id: edgeId,
            source: fromNodeId,
            target: toNodeId,
            label: data.label || '',
            weight: data.weight || 1.0,
            color: data.color || '#64748b',
            style: data.style || 'solid', // solid, dashed, dotted
            directed: data.directed !== undefined ? data.directed : this.properties.directed,
            properties: data.properties || {},
            created: new Date(),
            modified: new Date()
        };

        this.edges.set(edgeId, edge);
        this.emit('edgeAdded', { edge });
        this.emit('graphChanged', { type: 'edgeAdded', edge });
        
        return edge;
    }

    updateEdge(edgeId, updates) {
        const edge = this.edges.get(edgeId);
        if (!edge) {
            throw new Error(`Edge ${edgeId} not found`);
        }

        Object.assign(edge, updates);
        edge.modified = new Date();

        this.emit('edgeUpdated', { edge, updates });
        this.emit('graphChanged', { type: 'edgeUpdated', edge, updates });
        
        return edge;
    }

    deleteEdge(edgeId) {
        const edge = this.edges.get(edgeId);
        if (!edge) {
            throw new Error(`Edge ${edgeId} not found`);
        }

        this.edges.delete(edgeId);
        
        this.emit('edgeDeleted', { edgeId, edge });
        this.emit('graphChanged', { type: 'edgeDeleted', edgeId, edge });
        
        return edge;
    }

    getEdge(edgeId) {
        return this.edges.get(edgeId);
    }

    getAllEdges() {
        return Array.from(this.edges.values());
    }

    findEdge(fromNodeId, toNodeId) {
        return Array.from(this.edges.values()).find(edge => 
            edge.source === fromNodeId && edge.target === toNodeId
        );
    }

    // Graph analysis methods
    getNodeEdges(nodeId) {
        return Array.from(this.edges.values()).filter(edge => 
            edge.source === nodeId || edge.target === nodeId
        );
    }

    getNodeNeighbors(nodeId) {
        const edges = this.getNodeEdges(nodeId);
        const neighbors = new Set();
        
        edges.forEach(edge => {
            if (edge.source === nodeId) {
                neighbors.add(edge.target);
            }
            if (edge.target === nodeId && (!edge.directed || !this.properties.directed)) {
                neighbors.add(edge.source);
            }
        });
        
        return Array.from(neighbors).map(id => this.nodes.get(id));
    }

    getNodeDegree(nodeId) {
        return this.getNodeEdges(nodeId).length;
    }

    getGraphDensity() {
        const nodeCount = this.nodes.size;
        const edgeCount = this.edges.size;
        
        if (nodeCount < 2) return 0;
        
        const maxEdges = this.properties.directed 
            ? nodeCount * (nodeCount - 1)
            : nodeCount * (nodeCount - 1) / 2;
            
        return edgeCount / maxEdges;
    }

    // Advanced operations
    insertNodeBetween(edgeId, nodeData = {}) {
        const edge = this.edges.get(edgeId);
        if (!edge) {
            throw new Error(`Edge ${edgeId} not found`);
        }

        const sourceNode = this.nodes.get(edge.source);
        const targetNode = this.nodes.get(edge.target);
        
        // Calculate position for new node (midpoint)
        const newNodeData = {
            ...nodeData,
            x: (sourceNode.x + targetNode.x) / 2,
            y: (sourceNode.y + targetNode.y) / 2,
            label: nodeData.label || `Intermediate ${this.nodeIdCounter + 1}`
        };

        // Create the new node
        const newNode = this.createNode(newNodeData);

        // Delete the original edge
        this.deleteEdge(edgeId);

        // Create two new edges
        const edge1 = this.createEdge(edge.source, newNode.id, {
            label: edge.label,
            weight: edge.weight,
            color: edge.color,
            style: edge.style
        });

        const edge2 = this.createEdge(newNode.id, edge.target, {
            label: edge.label,
            weight: edge.weight,
            color: edge.color,
            style: edge.style
        });

        this.emit('nodeInserted', { 
            newNode, 
            originalEdge: edge, 
            newEdges: [edge1, edge2] 
        });

        return {
            node: newNode,
            edges: [edge1, edge2],
            originalEdge: edge
        };
    }

    createTriangle(edgeId, nodeData = {}) {
        const edge = this.edges.get(edgeId);
        if (!edge) {
            throw new Error(`Edge ${edgeId} not found`);
        }

        const sourceNode = this.nodes.get(edge.source);
        const targetNode = this.nodes.get(edge.target);

        // Calculate position for new node (forming a triangle)
        const angle = Math.atan2(targetNode.y - sourceNode.y, targetNode.x - sourceNode.x);
        const distance = Math.sqrt(
            Math.pow(targetNode.x - sourceNode.x, 2) + 
            Math.pow(targetNode.y - sourceNode.y, 2)
        );
        
        // Position the third node to form an equilateral triangle
        const triangleHeight = distance * Math.sin(Math.PI / 3);
        const newNodeData = {
            ...nodeData,
            x: (sourceNode.x + targetNode.x) / 2 - triangleHeight * Math.sin(angle),
            y: (sourceNode.y + targetNode.y) / 2 + triangleHeight * Math.cos(angle),
            label: nodeData.label || `Triangle ${this.nodeIdCounter + 1}`
        };

        // Create the new node
        const newNode = this.createNode(newNodeData);

        // Create edges to form triangle
        const edge1 = this.createEdge(edge.source, newNode.id, {
            label: nodeData.edgeLabel || 'CONNECTS',
            weight: 1.0,
            color: '#64748b'
        });

        const edge2 = this.createEdge(newNode.id, edge.target, {
            label: nodeData.edgeLabel || 'CONNECTS',
            weight: 1.0,
            color: '#64748b'
        });

        this.emit('triangleCreated', {
            newNode,
            existingEdge: edge,
            newEdges: [edge1, edge2]
        });

        return {
            node: newNode,
            edges: [edge1, edge2],
            existingEdge: edge
        };
    }

    // Graph algorithms
    findShortestPath(fromNodeId, toNodeId) {
        // Simple BFS implementation
        if (fromNodeId === toNodeId) return [fromNodeId];
        
        const visited = new Set();
        const queue = [[fromNodeId]];
        
        while (queue.length > 0) {
            const path = queue.shift();
            const currentNode = path[path.length - 1];
            
            if (visited.has(currentNode)) continue;
            visited.add(currentNode);
            
            const neighbors = this.getNodeNeighbors(currentNode);
            
            for (const neighbor of neighbors) {
                const newPath = [...path, neighbor.id];
                
                if (neighbor.id === toNodeId) {
                    return newPath;
                }
                
                if (!visited.has(neighbor.id)) {
                    queue.push(newPath);
                }
            }
        }
        
        return null; // No path found
    }

    findConnectedComponents() {
        const visited = new Set();
        const components = [];
        
        for (const [nodeId] of this.nodes) {
            if (!visited.has(nodeId)) {
                const component = [];
                const stack = [nodeId];
                
                while (stack.length > 0) {
                    const currentId = stack.pop();
                    
                    if (visited.has(currentId)) continue;
                    visited.add(currentId);
                    component.push(currentId);
                    
                    const neighbors = this.getNodeNeighbors(currentId);
                    neighbors.forEach(neighbor => {
                        if (!visited.has(neighbor.id)) {
                            stack.push(neighbor.id);
                        }
                    });
                }
                
                components.push(component);
            }
        }
        
        return components;
    }

    // Data operations
    clear() {
        const nodeCount = this.nodes.size;
        const edgeCount = this.edges.size;
        
        this.nodes.clear();
        this.edges.clear();
        this.nodeIdCounter = 0;
        this.edgeIdCounter = 0;
        
        this.emit('graphCleared', { nodeCount, edgeCount });
        this.emit('graphChanged', { type: 'cleared', nodeCount, edgeCount });
    }

    exportToJSON() {
        return {
            nodes: Array.from(this.nodes.values()),
            edges: Array.from(this.edges.values()),
            properties: this.properties,
            metadata: {
                nodeCount: this.nodes.size,
                edgeCount: this.edges.size,
                density: this.getGraphDensity(),
                exported: new Date().toISOString()
            }
        };
    }

    importFromJSON(data) {
        this.clear();
        
        // Import nodes
        if (data.nodes) {
            data.nodes.forEach(nodeData => {
                const node = { ...nodeData };
                this.nodes.set(node.id, node);
                
                // Update counter to avoid ID conflicts
                const idNum = parseInt(node.id.split('_')[1]);
                if (idNum > this.nodeIdCounter) {
                    this.nodeIdCounter = idNum;
                }
            });
        }
        
        // Import edges
        if (data.edges) {
            data.edges.forEach(edgeData => {
                const edge = { ...edgeData };
                this.edges.set(edge.id, edge);
                
                // Update counter to avoid ID conflicts
                const idNum = parseInt(edge.id.split('_')[1]);
                if (idNum > this.edgeIdCounter) {
                    this.edgeIdCounter = idNum;
                }
            });
        }
        
        // Import properties
        if (data.properties) {
            this.properties = { ...this.properties, ...data.properties };
        }
        
        this.emit('graphImported', data);
        this.emit('graphChanged', { type: 'imported', data });
    }

    // Utility methods
    getStatistics() {
        const components = this.findConnectedComponents();
        const degrees = Array.from(this.nodes.keys()).map(id => this.getNodeDegree(id));
        
        return {
            nodeCount: this.nodes.size,
            edgeCount: this.edges.size,
            density: this.getGraphDensity(),
            componentCount: components.length,
            averageDegree: degrees.length > 0 ? degrees.reduce((a, b) => a + b, 0) / degrees.length : 0,
            maxDegree: degrees.length > 0 ? Math.max(...degrees) : 0,
            minDegree: degrees.length > 0 ? Math.min(...degrees) : 0
        };
    }

    validateGraph() {
        const errors = [];
        
        // Check for orphaned edges
        this.edges.forEach(edge => {
            if (!this.nodes.has(edge.source)) {
                errors.push(`Edge ${edge.id} references non-existent source node ${edge.source}`);
            }
            if (!this.nodes.has(edge.target)) {
                errors.push(`Edge ${edge.id} references non-existent target node ${edge.target}`);
            }
        });
        
        return {
            valid: errors.length === 0,
            errors
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UniversalGraphEngine;
} else if (typeof window !== 'undefined') {
    window.UniversalGraphEngine = UniversalGraphEngine;
}