# Universal Graph Engine - Graphical User Interface

A modern, interactive web-based GUI for the Universal Graph Engine featuring smooth animations, intuitive controls, and advanced graph manipulation capabilities.

## 🚀 Features

### **Core Functionality**
- **Interactive Graph Visualization** - Smooth D3.js-powered animations with force-directed layout
- **Multiple Layout Algorithms** - Force-directed, circular, hierarchical, and grid layouts
- **Real-time Node/Edge Manipulation** - Create, edit, delete, and connect nodes with live updates
- **Advanced Graph Operations** - Insert nodes between edges, create triangular connections
- **Property Management** - Rich node properties with colors, sizes, descriptions, and custom data

### **User Interface**
- **Modern Design** - Clean, professional interface with smooth animations
- **Intuitive Tools** - Easy-to-use toolbar with select, node, edge, insert, and triangle tools
- **Context Menus** - Right-click nodes for quick actions and advanced operations
- **Modal Dialogs** - Beautiful modal windows for detailed editing and configuration
- **Responsive Layout** - Works on desktop, tablet, and mobile devices

### **Advanced Features**
- **Quick Edit Mode** - Single-click nodes for inline property editing
- **Expand Mode** - Double-click for detailed text/notes editing
- **Smart Connections** - Intelligent edge creation with visual feedback
- **Graph Statistics** - Live node count, edge count, and density calculations
- **Export/Import** - Save graphs as JSON or export as DOT files for Graphviz

### **Interactive Operations**
1. **Create Lonely Nodes** - Click anywhere with node tool to create isolated nodes
2. **Simple Connections** - Connect any two nodes with the edge tool
3. **Insert Between** - Click any edge to insert a node between two connected nodes
4. **Create Triangles** - Click any edge to create a third node connecting to both endpoints
5. **Quick Edit** - Click nodes to select and edit properties in real-time
6. **Expand Edit** - Right-click for context menu with advanced editing options

## 🎯 Quick Start

### **1. Open the Interface**
```bash
# Simply open index.html in a modern web browser
open index.html

# Or serve with a simple HTTP server
python -m http.server 8000
# Then visit http://localhost:8000
```

### **2. Basic Usage**

**Creating Nodes:**
- Select the Node tool (N key)
- Click anywhere on the canvas to create a node
- Each node gets a unique name and can be customized

**Connecting Nodes:**
- Select the Edge tool (E key)
- Click first node, then click second node
- A directed edge will be created between them

**Inserting Intermediate Nodes:**
- Select the Insert tool (I key)
- Click any existing edge
- A new node will be inserted between the connected nodes

**Creating Triangular Connections:**
- Select the Triangle tool (T key)
- Click any existing edge
- A new node will be created that connects to both endpoints

**Editing Nodes:**
- Click any node to select it and edit properties in the sidebar
- Right-click for context menu with advanced options
- Double-click to open detailed editing modal

## 📋 Interface Guide

### **Toolbar (Top Center)**
- **🖱️ Select Tool (S)** - Select, move, and edit nodes/edges
- **⭕ Node Tool (N)** - Create new nodes by clicking
- **➡️ Edge Tool (E)** - Connect nodes by clicking two nodes
- **➕ Insert Tool (I)** - Insert nodes between existing edges
- **▶️ Triangle Tool (T)** - Create triangular connections from edges
- **🔍 Zoom In (+)** - Zoom into the graph
- **🔍 Zoom Out (-)** - Zoom out of the graph
- **📐 Fit View (F)** - Fit all nodes in view

### **Sidebar (Left)**
**Graph Statistics:**
- Node count, edge count, and graph density
- Updates in real-time as you modify the graph

**Node Properties:**
- Label, description, color, and size controls
- Live preview of changes
- Custom property support

**Graph Controls:**
- Layout algorithm selection
- Animation speed control
- Clear graph button

### **Context Menu (Right-click)**
- **Edit Node** - Open detailed editing modal
- **Connect to...** - Create connections to other nodes
- **Insert Node Between** - Switch to insert mode
- **Create Triangle** - Switch to triangle mode
- **Delete Node** - Remove node and all connections

### **Keyboard Shortcuts**
- **S** - Select tool
- **N** - Node tool
- **E** - Edge tool
- **I** - Insert tool
- **T** - Triangle tool
- **F** - Fit to view
- **+/=** - Zoom in
- **-** - Zoom out
- **Delete/Backspace** - Delete selected items
- **Escape** - Clear selection or close modals

## 🎨 Customization

### **Node Styling**
```javascript
// Nodes support full customization
{
    label: "Custom Node",
    description: "Detailed description",
    color: "#3b82f6",
    size: 50,
    properties: {
        category: "Important",
        priority: "High"
    }
}
```

### **Edge Styling**
```javascript
// Edges can be customized with labels and weights
{
    label: "RELATIONSHIP_TYPE",
    weight: 0.8,
    color: "#64748b",
    style: "solid" // solid, dashed, dotted
}
```

### **Layout Algorithms**
- **Force-Directed** - Physics-based natural layout
- **Circular** - Nodes arranged in a circle
- **Hierarchical** - Tree-like structure with levels
- **Grid** - Regular grid arrangement

## 📊 Graph Operations

### **Basic Operations**
- **Create Node** - `engine.createNode(data)`
- **Create Edge** - `engine.createEdge(fromId, toId, data)`
- **Update Node** - `engine.updateNode(nodeId, updates)`
- **Delete Node** - `engine.deleteNode(nodeId)`

### **Advanced Operations**
- **Insert Between** - `engine.insertNodeBetween(edgeId, nodeData)`
- **Create Triangle** - `engine.createTriangle(edgeId, nodeData)`
- **Find Path** - `engine.findShortestPath(fromId, toId)`
- **Graph Analysis** - `engine.getStatistics()`

### **Data Operations**
- **Export JSON** - `engine.exportToJSON()`
- **Import JSON** - `engine.importFromJSON(data)`
- **Clear All** - `engine.clear()`
- **Validate** - `engine.validateGraph()`

## 🔧 Technical Details

### **Architecture**
```
📁 gui/
├── 📄 index.html          # Main HTML structure
├── 🎨 styles.css          # Modern CSS styling
├── 🔧 graph-engine.js     # Core graph data management
├── 🖼️ graph-ui.js         # D3.js visualization and UI
├── 📱 app.js              # Main application controller
└── 📖 README.md           # This documentation
```

### **Dependencies**
- **D3.js v7** - Data visualization and DOM manipulation
- **Font Awesome 6** - Icons and UI elements
- **Inter Font** - Modern typography
- **Modern Browser** - ES6+ support required

### **Browser Support**
- Chrome 80+ ✅
- Firefox 75+ ✅
- Safari 13+ ✅
- Edge 80+ ✅

## 🎯 Use Cases

### **Social Networks**
```javascript
// Create people and relationships
const alice = engine.createNode({label: "Alice", description: "Software Engineer"});
const bob = engine.createNode({label: "Bob", description: "Data Scientist"});
engine.createEdge(alice.id, bob.id, {label: "COLLEAGUES"});
```

### **Knowledge Graphs**
```javascript
// Model concepts and relationships
const math = engine.createNode({label: "Mathematics", color: "#3b82f6"});
const stats = engine.createNode({label: "Statistics", color: "#10b981"});
engine.createEdge(math.id, stats.id, {label: "FOUNDATION_OF"});
```

### **Workflow Diagrams**
```javascript
// Process flows with intermediate steps
const start = engine.createNode({label: "Start Process"});
const end = engine.createNode({label: "End Process"});
const startToEnd = engine.createEdge(start.id, end.id, {label: "PROCESS"});

// Insert validation step
engine.insertNodeBetween(startToEnd.id, {
    label: "Validation",
    description: "Data validation step"
});
```

### **Organizational Charts**
```javascript
// Hierarchical structures with cross-connections
const ceo = engine.createNode({label: "CEO"});
const cto = engine.createNode({label: "CTO"});
const dev = engine.createNode({label: "Developer"});

engine.createEdge(ceo.id, cto.id, {label: "MANAGES"});
engine.createEdge(cto.id, dev.id, {label: "MANAGES"});

// Add cross-functional connection
engine.createTriangle(cto_to_dev_edge.id, {
    label: "Project Manager",
    description: "Coordinates between CTO and Developer"
});
```

## 🚀 Advanced Features

### **Real-time Collaboration**
```javascript
// Engine supports event-driven updates
engine.addEventListener('nodeAdded', (data) => {
    // Broadcast to other users
    socket.emit('nodeAdded', data);
});
```

### **Graph Algorithms**
```javascript
// Built-in graph analysis
const stats = engine.getStatistics();
const path = engine.findShortestPath(nodeA.id, nodeB.id);
const components = engine.findConnectedComponents();
```

### **Custom Layouts**
```javascript
// Implement custom positioning algorithms
function applyCustomLayout(nodes) {
    nodes.forEach((node, index) => {
        engine.updateNode(node.id, {
            x: customX(index),
            y: customY(index)
        });
    });
}
```

### **Data Integration**
```javascript
// Import from various sources
fetch('/api/graph-data')
    .then(response => response.json())
    .then(data => engine.importFromJSON(data));
```

## 🎉 Examples

The interface includes several demonstration scenarios:

1. **Team Collaboration Network** - Shows how team members interact
2. **Project Dependencies** - Visualizes task relationships
3. **Knowledge Base** - Connected concepts and topics
4. **Social Connections** - Friend and family networks

## 🔮 Future Enhancements

- **3D Visualization** - Three.js integration for 3D graphs
- **Real-time Collaboration** - Multi-user editing with WebSockets
- **Advanced Analytics** - Community detection, centrality measures
- **Plugin System** - Custom tools and visualizations
- **Mobile App** - Native mobile interface
- **AR/VR Support** - Immersive graph exploration

---

**Ready to explore complex relationships visually?** Open `index.html` in your browser and start creating your first graph! 🚀