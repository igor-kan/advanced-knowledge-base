/**
 * Universal Graph Engine - Persistence Manager
 * 
 * Handles data persistence across browser sessions using multiple storage methods:
 * - Local Storage for automatic saving
 * - IndexedDB for large graphs
 * - File system for manual export/import
 * - Server sync for multi-device access
 */

class PersistenceManager {
    constructor(engine) {
        this.engine = engine;
        this.settings = {
            autoSave: true,
            autoSaveInterval: 30000, // 30 seconds
            maxLocalStorageSize: 5 * 1024 * 1024, // 5MB
            compressionEnabled: true,
            versioning: true,
            maxVersions: 10
        };
        
        this.storage = {
            localStorage: window.localStorage,
            indexedDB: null,
            isIndexedDBAvailable: false
        };
        
        this.autoSaveTimer = null;
        this.lastSaved = null;
        this.currentVersion = 1;
        
        this.init();
    }

    async init() {
        console.log('🔄 Initializing Persistence Manager...');
        
        // Initialize IndexedDB
        await this.initIndexedDB();
        
        // Load settings
        this.loadSettings();
        
        // Load last saved graph
        await this.loadLastGraph();
        
        // Start auto-save if enabled
        if (this.settings.autoSave) {
            this.startAutoSave();
        }
        
        // Listen for graph changes
        this.bindGraphEvents();
        
        // Listen for page unload
        this.bindPageEvents();
        
        console.log('✅ Persistence Manager initialized');
    }

    async initIndexedDB() {
        try {
            const request = indexedDB.open('UniversalGraphEngine', 1);
            
            request.onerror = () => {
                console.warn('⚠️ IndexedDB not available, using localStorage only');
                this.storage.isIndexedDBAvailable = false;
            };
            
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                
                // Create graphs store
                if (!db.objectStoreNames.contains('graphs')) {
                    const graphStore = db.createObjectStore('graphs', { keyPath: 'id' });
                    graphStore.createIndex('name', 'name', { unique: false });
                    graphStore.createIndex('created', 'created', { unique: false });
                    graphStore.createIndex('modified', 'modified', { unique: false });
                }
                
                // Create versions store
                if (!db.objectStoreNames.contains('versions')) {
                    const versionStore = db.createObjectStore('versions', { keyPath: 'id' });
                    versionStore.createIndex('graphId', 'graphId', { unique: false });
                    versionStore.createIndex('created', 'created', { unique: false });
                }
                
                // Create settings store
                if (!db.objectStoreNames.contains('settings')) {
                    db.createObjectStore('settings', { keyPath: 'key' });
                }
            };
            
            return new Promise((resolve, reject) => {
                request.onsuccess = (event) => {
                    this.storage.indexedDB = event.target.result;
                    this.storage.isIndexedDBAvailable = true;
                    console.log('✅ IndexedDB initialized');
                    resolve(this.storage.indexedDB);
                };
                
                request.onerror = () => {
                    console.warn('⚠️ Failed to initialize IndexedDB');
                    this.storage.isIndexedDBAvailable = false;
                    reject(request.error);
                };
            });
        } catch (error) {
            console.warn('⚠️ IndexedDB not supported:', error);
            this.storage.isIndexedDBAvailable = false;
        }
    }

    // Auto-save functionality
    startAutoSave() {
        if (this.autoSaveTimer) {
            clearInterval(this.autoSaveTimer);
        }
        
        this.autoSaveTimer = setInterval(() => {
            this.autoSaveGraph();
        }, this.settings.autoSaveInterval);
        
        console.log(`🔄 Auto-save started (${this.settings.autoSaveInterval / 1000}s interval)`);
    }

    stopAutoSave() {
        if (this.autoSaveTimer) {
            clearInterval(this.autoSaveTimer);
            this.autoSaveTimer = null;
            console.log('⏹️ Auto-save stopped');
        }
    }

    async autoSaveGraph() {
        try {
            await this.saveGraph('__autosave__', {
                name: 'Auto-saved Graph',
                description: 'Automatically saved graph',
                isAutoSave: true
            });
            
            this.lastSaved = new Date();
            this.updateSaveStatus('Auto-saved');
            
        } catch (error) {
            console.error('❌ Auto-save failed:', error);
            this.updateSaveStatus('Auto-save failed', 'error');
        }
    }

    // Core persistence methods
    async saveGraph(graphId = null, metadata = {}) {
        const graphData = this.engine.exportToJSON();
        const saveData = {
            id: graphId || this.generateGraphId(),
            name: metadata.name || `Graph ${new Date().toLocaleString()}`,
            description: metadata.description || '',
            graphData: graphData,
            created: metadata.created || new Date().toISOString(),
            modified: new Date().toISOString(),
            version: this.currentVersion++,
            size: JSON.stringify(graphData).length,
            isAutoSave: metadata.isAutoSave || false,
            checksum: this.calculateChecksum(graphData)
        };

        // Compress if enabled and data is large
        if (this.settings.compressionEnabled && saveData.size > 100000) {
            saveData.graphData = this.compressData(saveData.graphData);
            saveData.compressed = true;
        }

        // Save to appropriate storage
        if (this.storage.isIndexedDBAvailable && saveData.size > 100000) {
            await this.saveToIndexedDB(saveData);
        } else {
            await this.saveToLocalStorage(saveData);
        }

        // Save version history if enabled
        if (this.settings.versioning && !metadata.isAutoSave) {
            await this.saveVersion(saveData);
        }

        console.log(`💾 Graph saved: ${saveData.name} (${this.formatSize(saveData.size)})`);
        return saveData.id;
    }

    async loadGraph(graphId) {
        let graphData = null;

        // Try IndexedDB first for large graphs
        if (this.storage.isIndexedDBAvailable) {
            graphData = await this.loadFromIndexedDB(graphId);
        }

        // Fallback to localStorage
        if (!graphData) {
            graphData = await this.loadFromLocalStorage(graphId);
        }

        if (!graphData) {
            throw new Error(`Graph ${graphId} not found`);
        }

        // Decompress if needed
        if (graphData.compressed) {
            graphData.graphData = this.decompressData(graphData.graphData);
        }

        // Verify integrity
        const currentChecksum = this.calculateChecksum(graphData.graphData);
        if (graphData.checksum && graphData.checksum !== currentChecksum) {
            console.warn('⚠️ Graph data integrity check failed');
        }

        // Load into engine
        this.engine.importFromJSON(graphData.graphData);
        
        console.log(`📂 Graph loaded: ${graphData.name}`);
        return graphData;
    }

    async loadLastGraph() {
        try {
            // Try to load auto-saved graph first
            const autoSave = await this.loadGraph('__autosave__');
            if (autoSave && autoSave.graphData.nodes.length > 0) {
                console.log('📂 Loaded auto-saved graph');
                this.updateSaveStatus('Loaded from auto-save');
                return autoSave;
            }
        } catch (error) {
            // Auto-save not found, that's okay
        }

        try {
            // Try to load the most recent manual save
            const recentGraphs = await this.listGraphs();
            const manualSaves = recentGraphs.filter(g => !g.isAutoSave);
            
            if (manualSaves.length > 0) {
                const mostRecent = manualSaves.sort((a, b) => 
                    new Date(b.modified) - new Date(a.modified)
                )[0];
                
                const loaded = await this.loadGraph(mostRecent.id);
                console.log(`📂 Loaded most recent graph: ${loaded.name}`);
                this.updateSaveStatus('Loaded previous session');
                return loaded;
            }
        } catch (error) {
            console.error('❌ Failed to load last graph:', error);
        }

        console.log('ℹ️ No previous graph found, starting fresh');
        return null;
    }

    // Storage implementations
    async saveToLocalStorage(saveData) {
        try {
            const key = `uge_graph_${saveData.id}`;
            const serialized = JSON.stringify(saveData);
            
            // Check size limits
            if (serialized.length > this.settings.maxLocalStorageSize) {
                throw new Error('Graph too large for localStorage');
            }
            
            this.storage.localStorage.setItem(key, serialized);
            
            // Update index
            await this.updateStorageIndex(saveData, 'localStorage');
            
        } catch (error) {
            if (error.name === 'QuotaExceededError') {
                await this.cleanupOldGraphs('localStorage');
                // Retry once
                this.storage.localStorage.setItem(`uge_graph_${saveData.id}`, JSON.stringify(saveData));
            } else {
                throw error;
            }
        }
    }

    async loadFromLocalStorage(graphId) {
        try {
            const key = `uge_graph_${graphId}`;
            const serialized = this.storage.localStorage.getItem(key);
            
            if (!serialized) {
                return null;
            }
            
            return JSON.parse(serialized);
        } catch (error) {
            console.error('❌ Failed to load from localStorage:', error);
            return null;
        }
    }

    async saveToIndexedDB(saveData) {
        if (!this.storage.isIndexedDBAvailable) {
            throw new Error('IndexedDB not available');
        }

        return new Promise((resolve, reject) => {
            const transaction = this.storage.indexedDB.transaction(['graphs'], 'readwrite');
            const store = transaction.objectStore('graphs');
            const request = store.put(saveData);

            request.onsuccess = () => {
                this.updateStorageIndex(saveData, 'indexedDB');
                resolve(request.result);
            };

            request.onerror = () => reject(request.error);
        });
    }

    async loadFromIndexedDB(graphId) {
        if (!this.storage.isIndexedDBAvailable) {
            return null;
        }

        return new Promise((resolve, reject) => {
            const transaction = this.storage.indexedDB.transaction(['graphs'], 'readonly');
            const store = transaction.objectStore('graphs');
            const request = store.get(graphId);

            request.onsuccess = () => resolve(request.result || null);
            request.onerror = () => reject(request.error);
        });
    }

    // Graph management
    async listGraphs() {
        const graphs = [];

        // Get from localStorage
        for (let i = 0; i < this.storage.localStorage.length; i++) {
            const key = this.storage.localStorage.key(i);
            if (key.startsWith('uge_graph_')) {
                try {
                    const data = JSON.parse(this.storage.localStorage.getItem(key));
                    graphs.push({
                        id: data.id,
                        name: data.name,
                        description: data.description,
                        created: data.created,
                        modified: data.modified,
                        size: data.size,
                        isAutoSave: data.isAutoSave,
                        storage: 'localStorage'
                    });
                } catch (error) {
                    console.warn(`⚠️ Failed to parse graph ${key}:`, error);
                }
            }
        }

        // Get from IndexedDB
        if (this.storage.isIndexedDBAvailable) {
            try {
                const indexedDBGraphs = await this.getIndexedDBGraphs();
                graphs.push(...indexedDBGraphs);
            } catch (error) {
                console.error('❌ Failed to load IndexedDB graphs:', error);
            }
        }

        // Remove duplicates and sort
        const uniqueGraphs = graphs.reduce((acc, graph) => {
            const existing = acc.find(g => g.id === graph.id);
            if (!existing || new Date(graph.modified) > new Date(existing.modified)) {
                acc = acc.filter(g => g.id !== graph.id);
                acc.push(graph);
            }
            return acc;
        }, []);

        return uniqueGraphs.sort((a, b) => new Date(b.modified) - new Date(a.modified));
    }

    async getIndexedDBGraphs() {
        return new Promise((resolve, reject) => {
            const transaction = this.storage.indexedDB.transaction(['graphs'], 'readonly');
            const store = transaction.objectStore('graphs');
            const request = store.getAll();

            request.onsuccess = () => {
                const graphs = request.result.map(data => ({
                    id: data.id,
                    name: data.name,
                    description: data.description,
                    created: data.created,
                    modified: data.modified,
                    size: data.size,
                    isAutoSave: data.isAutoSave,
                    storage: 'indexedDB'
                }));
                resolve(graphs);
            };

            request.onerror = () => reject(request.error);
        });
    }

    async deleteGraph(graphId) {
        // Delete from localStorage
        this.storage.localStorage.removeItem(`uge_graph_${graphId}`);

        // Delete from IndexedDB
        if (this.storage.isIndexedDBAvailable) {
            return new Promise((resolve, reject) => {
                const transaction = this.storage.indexedDB.transaction(['graphs'], 'readwrite');
                const store = transaction.objectStore('graphs');
                const request = store.delete(graphId);

                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });
        }

        console.log(`🗑️ Graph deleted: ${graphId}`);
    }

    // Version management
    async saveVersion(graphData) {
        if (!this.settings.versioning) return;

        const versionData = {
            id: `${graphData.id}_v${graphData.version}`,
            graphId: graphData.id,
            version: graphData.version,
            created: new Date().toISOString(),
            graphData: graphData.graphData,
            size: graphData.size,
            checksum: graphData.checksum
        };

        if (this.storage.isIndexedDBAvailable) {
            const transaction = this.storage.indexedDB.transaction(['versions'], 'readwrite');
            const store = transaction.objectStore('versions');
            store.put(versionData);
        }

        // Cleanup old versions
        await this.cleanupOldVersions(graphData.id);
    }

    async getVersionHistory(graphId) {
        if (!this.storage.isIndexedDBAvailable) return [];

        return new Promise((resolve, reject) => {
            const transaction = this.storage.indexedDB.transaction(['versions'], 'readonly');
            const store = transaction.objectStore('versions');
            const index = store.index('graphId');
            const request = index.getAll(graphId);

            request.onsuccess = () => {
                const versions = request.result.sort((a, b) => b.version - a.version);
                resolve(versions);
            };

            request.onerror = () => reject(request.error);
        });
    }

    // Utility methods
    generateGraphId() {
        return `graph_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    calculateChecksum(data) {
        // Simple checksum for data integrity
        const str = JSON.stringify(data);
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return hash.toString(36);
    }

    compressData(data) {
        // Simple compression using JSON minification
        // In a real implementation, you might use LZ-string or similar
        return JSON.stringify(data);
    }

    decompressData(compressedData) {
        return JSON.parse(compressedData);
    }

    formatSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Cleanup methods
    async cleanupOldGraphs(storageType) {
        const graphs = await this.listGraphs();
        const autoSaves = graphs.filter(g => g.isAutoSave && g.storage === storageType);
        
        // Keep only the most recent auto-save
        if (autoSaves.length > 1) {
            const sorted = autoSaves.sort((a, b) => new Date(b.modified) - new Date(a.modified));
            const toDelete = sorted.slice(1);
            
            for (const graph of toDelete) {
                await this.deleteGraph(graph.id);
            }
        }
    }

    async cleanupOldVersions(graphId) {
        const versions = await this.getVersionHistory(graphId);
        
        if (versions.length > this.settings.maxVersions) {
            const toDelete = versions.slice(this.settings.maxVersions);
            
            const transaction = this.storage.indexedDB.transaction(['versions'], 'readwrite');
            const store = transaction.objectStore('versions');
            
            toDelete.forEach(version => {
                store.delete(version.id);
            });
        }
    }

    // Settings management
    loadSettings() {
        try {
            const saved = this.storage.localStorage.getItem('uge_settings');
            if (saved) {
                const settings = JSON.parse(saved);
                this.settings = { ...this.settings, ...settings };
            }
        } catch (error) {
            console.warn('⚠️ Failed to load settings:', error);
        }
    }

    saveSettings() {
        try {
            this.storage.localStorage.setItem('uge_settings', JSON.stringify(this.settings));
        } catch (error) {
            console.error('❌ Failed to save settings:', error);
        }
    }

    updateSettings(newSettings) {
        this.settings = { ...this.settings, ...newSettings };
        this.saveSettings();
        
        // Restart auto-save if interval changed
        if (newSettings.autoSave !== undefined || newSettings.autoSaveInterval !== undefined) {
            if (this.settings.autoSave) {
                this.startAutoSave();
            } else {
                this.stopAutoSave();
            }
        }
    }

    // Event handlers
    bindGraphEvents() {
        this.engine.addEventListener('graphChanged', () => {
            this.updateSaveStatus('Unsaved changes', 'warning');
        });
    }

    bindPageEvents() {
        // Save before page unload
        window.addEventListener('beforeunload', (event) => {
            if (this.hasUnsavedChanges()) {
                this.autoSaveGraph();
                event.preventDefault();
                event.returnValue = 'You have unsaved changes. Are you sure you want to leave?';
            }
        });

        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                // Page is being hidden, save current state
                this.autoSaveGraph();
            }
        });
    }

    // Status management
    updateSaveStatus(message, type = 'info') {
        const statusElement = document.getElementById('save-status');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `save-status ${type}`;
        }
        
        // Show toast notification
        if (window.showToast) {
            window.showToast(message, type);
        }
    }

    hasUnsavedChanges() {
        // Simple check - in a real implementation, you'd track changes more precisely
        return this.lastSaved && (Date.now() - this.lastSaved.getTime()) > this.settings.autoSaveInterval;
    }

    // Public API
    async exportGraph(filename) {
        const graphData = this.engine.exportToJSON();
        const blob = new Blob([JSON.stringify(graphData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = filename || `graph_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
        link.click();
        
        URL.revokeObjectURL(url);
        console.log(`📤 Graph exported: ${link.download}`);
    }

    async importGraph(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                try {
                    const data = JSON.parse(event.target.result);
                    this.engine.importFromJSON(data);
                    console.log('📥 Graph imported from file');
                    resolve(data);
                } catch (error) {
                    reject(new Error('Invalid graph file format'));
                }
            };
            
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    // Storage info
    getStorageInfo() {
        const info = {
            localStorage: {
                available: !!this.storage.localStorage,
                used: 0,
                total: 5 * 1024 * 1024 // Approximate limit
            },
            indexedDB: {
                available: this.storage.isIndexedDBAvailable,
                used: 0,
                total: Number.MAX_SAFE_INTEGER // Usually limited by available disk space
            }
        };

        // Calculate localStorage usage
        if (this.storage.localStorage) {
            let totalSize = 0;
            for (let key in this.storage.localStorage) {
                if (this.storage.localStorage.hasOwnProperty(key)) {
                    totalSize += this.storage.localStorage[key].length;
                }
            }
            info.localStorage.used = totalSize;
        }

        return info;
    }

    async updateStorageIndex(saveData, storageType) {
        // Update storage index for efficient querying
        const indexKey = 'uge_storage_index';
        try {
            const existing = JSON.parse(this.storage.localStorage.getItem(indexKey) || '{}');
            existing[saveData.id] = {
                name: saveData.name,
                modified: saveData.modified,
                size: saveData.size,
                storage: storageType
            };
            this.storage.localStorage.setItem(indexKey, JSON.stringify(existing));
        } catch (error) {
            console.warn('⚠️ Failed to update storage index:', error);
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PersistenceManager;
} else if (typeof window !== 'undefined') {
    window.PersistenceManager = PersistenceManager;
}