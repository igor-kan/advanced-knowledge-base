/**
 * Comprehensive Knowledge Base Benchmark Suite
 * 
 * Performance testing for billions-scale knowledge base operations:
 * - Node creation and retrieval benchmarks
 * - Edge creation and traversal performance
 * - Complex pattern matching benchmarks
 * - Hypergraph operation performance
 * - Memory usage and scalability tests
 * - Concurrent operation benchmarks
 */

const { performance } = require('perf_hooks');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const fs = require('fs').promises;
const path = require('path');

// Knowledge Base Implementations
const Neo4jKnowledgeBase = require('../implementations/neo4j-graph/src/KnowledgeBase');
const RedisGraphKB = require('../implementations/redis-graph/src/RedisGraphKB');
const HypergraphKB = require('../implementations/hypergraph/src/HypergraphKB');
const CustomKnowledgeEngine = require('../implementations/custom-engine/src/CustomKnowledgeEngine');

class ComprehensiveBenchmark {
  constructor(config = {}) {
    this.config = {
      implementations: config.implementations || ['neo4j', 'redis', 'hypergraph', 'custom'],
      testSizes: config.testSizes || [1000, 10000, 100000, 1000000],
      concurrencyLevels: config.concurrencyLevels || [1, 10, 50, 100],
      iterations: config.iterations || 3,
      warmupRuns: config.warmupRuns || 1,
      outputDir: config.outputDir || './benchmark-results',
      enableMemoryProfiling: config.enableMemoryProfiling !== false,
      enableCpuProfiling: config.enableCpuProfiling || false,
      timeout: config.timeout || 300000, // 5 minutes
      ...config
    };
    
    this.knowledgeBases = new Map();
    this.results = new Map();
    this.currentTest = null;
    
    // Test data generators
    this.dataGenerators = {
      nodes: new NodeDataGenerator(),
      edges: new EdgeDataGenerator(),
      patterns: new PatternDataGenerator(),
      hyperedges: new HyperedgeDataGenerator()
    };
    
    // Performance monitors
    this.memoryMonitor = new MemoryMonitor();
    this.cpuMonitor = new CpuMonitor();
    this.networkMonitor = new NetworkMonitor();
    
    // Statistics collector
    this.statistics = new StatisticsCollector();
  }

  /**
   * Run comprehensive benchmark suite
   */
  async runBenchmarks() {
    console.log('🚀 Starting Comprehensive Knowledge Base Benchmarks');
    console.log('=' * 60);
    
    try {
      // Initialize implementations
      await this.initializeImplementations();
      
      // Create output directory
      await fs.mkdir(this.config.outputDir, { recursive: true });
      
      // Run benchmark categories
      const benchmarkSuites = [
        { name: 'Node Operations', fn: this.benchmarkNodeOperations.bind(this) },
        { name: 'Edge Operations', fn: this.benchmarkEdgeOperations.bind(this) },
        { name: 'Pattern Matching', fn: this.benchmarkPatternMatching.bind(this) },
        { name: 'Graph Traversal', fn: this.benchmarkGraphTraversal.bind(this) },
        { name: 'Hypergraph Operations', fn: this.benchmarkHypergraphOperations.bind(this) },
        { name: 'Batch Operations', fn: this.benchmarkBatchOperations.bind(this) },
        { name: 'Concurrent Operations', fn: this.benchmarkConcurrentOperations.bind(this) },
        { name: 'Memory Scalability', fn: this.benchmarkMemoryScalability.bind(this) },
        { name: 'Query Performance', fn: this.benchmarkQueryPerformance.bind(this) },
        { name: 'Index Performance', fn: this.benchmarkIndexPerformance.bind(this) }
      ];
      
      for (const suite of benchmarkSuites) {
        console.log(`\n📊 Running ${suite.name} Benchmarks...`);
        await this.runBenchmarkSuite(suite.name, suite.fn);
      }
      
      // Generate comprehensive report
      await this.generateReport();
      
      console.log('\n✅ Benchmark suite completed successfully!');
      console.log(`📄 Results saved to: ${this.config.outputDir}`);
      
    } catch (error) {
      console.error('❌ Benchmark suite failed:', error);
      throw error;
    } finally {
      // Cleanup
      await this.cleanup();
    }
  }

  /**
   * Initialize all knowledge base implementations
   */
  async initializeImplementations() {
    console.log('📚 Initializing knowledge base implementations...');
    
    const implementations = {
      neo4j: async () => {
        const kb = new Neo4jKnowledgeBase({
          uri: process.env.NEO4J_URI || 'bolt://localhost:7687',
          username: process.env.NEO4J_USER || 'neo4j',
          password: process.env.NEO4J_PASSWORD || 'benchmark123'
        });
        await kb.connect();
        return kb;
      },
      
      redis: async () => {
        const kb = new RedisGraphKB({
          host: process.env.REDIS_HOST || 'localhost',
          port: process.env.REDIS_PORT || 6379,
          database: 1 // Use separate database for benchmarks
        });
        await kb.connect();
        return kb;
      },
      
      hypergraph: async () => {
        const kb = new HypergraphKB({
          maxHyperedgeSize: 10000,
          enableCompression: true,
          cacheSize: 100000
        });
        return kb;
      },
      
      custom: async () => {
        const kb = new CustomKnowledgeEngine({
          storageDir: './benchmark-data/custom',
          maxMemoryMB: 8192,
          enableParallelProcessing: true,
          workerCount: require('os').cpus().length
        });
        await kb.initialize();
        return kb;
      }
    };
    
    for (const impl of this.config.implementations) {
      if (implementations[impl]) {
        try {
          console.log(`  Initializing ${impl}...`);
          const kb = await implementations[impl]();
          this.knowledgeBases.set(impl, kb);
          console.log(`  ✅ ${impl} ready`);
        } catch (error) {
          console.warn(`  ⚠️  Failed to initialize ${impl}:`, error.message);
        }
      }
    }
    
    if (this.knowledgeBases.size === 0) {
      throw new Error('No knowledge base implementations available for benchmarking');
    }
  }

  /**
   * Run a benchmark suite
   */
  async runBenchmarkSuite(suiteName, benchmarkFn) {
    const suiteResults = new Map();
    
    for (const [implName, kb] of this.knowledgeBases) {
      console.log(`  Testing ${implName}...`);
      
      try {
        // Warmup
        for (let i = 0; i < this.config.warmupRuns; i++) {
          await this.runWarmup(kb, implName);
        }
        
        // Start monitoring
        this.memoryMonitor.start();
        this.cpuMonitor.start();
        
        // Run benchmark
        const results = await benchmarkFn(kb, implName);
        
        // Stop monitoring
        const memoryStats = this.memoryMonitor.stop();
        const cpuStats = this.cpuMonitor.stop();
        
        // Combine results
        suiteResults.set(implName, {
          ...results,
          memory: memoryStats,
          cpu: cpuStats,
          timestamp: new Date().toISOString()
        });
        
        console.log(`    ✅ ${implName} completed`);
        
      } catch (error) {
        console.error(`    ❌ ${implName} failed:`, error.message);
        suiteResults.set(implName, {
          error: error.message,
          timestamp: new Date().toISOString()
        });
      }
      
      // Cleanup between implementations
      await this.cleanupImplementation(kb);
    }
    
    this.results.set(suiteName, suiteResults);
  }

  /**
   * Benchmark node operations
   */
  async benchmarkNodeOperations(kb, implName) {
    const results = {
      creation: {},
      retrieval: {},
      update: {},
      deletion: {}
    };
    
    for (const size of this.config.testSizes) {
      console.log(`    Testing with ${size} nodes...`);
      
      // Node creation benchmark
      const nodeData = this.dataGenerators.nodes.generate(size);
      const creationResults = await this.benchmarkOperation(
        `${implName}_node_creation_${size}`,
        async () => {
          const promises = nodeData.map(data => kb.createNode(data));
          return await Promise.all(promises);
        }
      );
      
      results.creation[size] = creationResults;
      
      // Node retrieval benchmark
      const nodeIds = creationResults.result?.map(node => node.id) || [];
      const retrievalResults = await this.benchmarkOperation(
        `${implName}_node_retrieval_${size}`,
        async () => {
          const promises = nodeIds.slice(0, Math.min(1000, nodeIds.length))
            .map(id => kb.getNode(id));
          return await Promise.all(promises);
        }
      );
      
      results.retrieval[size] = retrievalResults;
      
      // Node update benchmark
      const updateResults = await this.benchmarkOperation(
        `${implName}_node_update_${size}`,
        async () => {
          const updatePromises = nodeIds.slice(0, Math.min(100, nodeIds.length))
            .map(id => kb.updateNode ? kb.updateNode(id, { 
              properties: { benchmarkUpdate: true, timestamp: Date.now() }
            }) : Promise.resolve());
          return await Promise.all(updatePromises);
        }
      );
      
      results.update[size] = updateResults;
      
      // Memory usage after operations
      results.creation[size].memoryAfter = process.memoryUsage();
    }
    
    return results;
  }

  /**
   * Benchmark edge operations
   */
  async benchmarkEdgeOperations(kb, implName) {
    const results = {
      creation: {},
      retrieval: {},
      traversal: {}
    };
    
    // First create nodes to connect
    const baseNodes = await this.createBaseNodes(kb, 1000);
    
    for (const size of this.config.testSizes.filter(s => s <= 100000)) {
      console.log(`    Testing with ${size} edges...`);
      
      // Edge creation benchmark
      const edgeData = this.dataGenerators.edges.generate(size, baseNodes);
      const creationResults = await this.benchmarkOperation(
        `${implName}_edge_creation_${size}`,
        async () => {
          const promises = edgeData.map(data => kb.createEdge(data));
          return await Promise.all(promises);
        }
      );
      
      results.creation[size] = creationResults;
      
      // Edge traversal benchmark
      const traversalResults = await this.benchmarkOperation(
        `${implName}_edge_traversal_${size}`,
        async () => {
          const startNode = baseNodes[0];
          return await kb.traverse(startNode.id, { 
            maxDepth: 3, 
            limit: 1000 
          });
        }
      );
      
      results.traversal[size] = traversalResults;
    }
    
    return results;
  }

  /**
   * Benchmark pattern matching
   */
  async benchmarkPatternMatching(kb, implName) {
    const results = {
      simple: {},
      complex: {},
      constraints: {}
    };
    
    // Create test graph
    await this.createTestGraph(kb, 10000, 50000);
    
    // Simple pattern matching
    const simplePattern = {
      nodes: {
        A: { type: 'Person' },
        B: { type: 'Company' }
      },
      edges: [
        { from: 'A', to: 'B', type: 'WORKS_FOR' }
      ]
    };
    
    const simpleResults = await this.benchmarkOperation(
      `${implName}_simple_pattern`,
      async () => {
        return await kb.findPattern(simplePattern);
      }
    );
    
    results.simple = simpleResults;
    
    // Complex pattern matching
    const complexPattern = {
      nodes: {
        A: { type: 'Person' },
        B: { type: 'Company' },
        C: { type: 'Project' },
        D: { type: 'Technology' }
      },
      edges: [
        { from: 'A', to: 'B', type: 'WORKS_FOR' },
        { from: 'A', to: 'C', type: 'ASSIGNED_TO' },
        { from: 'C', to: 'D', type: 'USES_TECHNOLOGY' }
      ]
    };
    
    const complexResults = await this.benchmarkOperation(
      `${implName}_complex_pattern`,
      async () => {
        return await kb.findPattern(complexPattern);
      }
    );
    
    results.complex = complexResults;
    
    return results;
  }

  /**
   * Benchmark hypergraph operations
   */
  async benchmarkHypergraphOperations(kb, implName) {
    if (!kb.createHyperedge) {
      return { message: 'Hypergraph operations not supported' };
    }
    
    const results = {
      creation: {},
      hyperpatterns: {},
      clustering: {}
    };
    
    // Create base nodes
    const baseNodes = await this.createBaseNodes(kb, 1000);
    
    for (const size of [100, 1000, 10000]) {
      console.log(`    Testing hypergraphs with ${size} hyperedges...`);
      
      // Hyperedge creation
      const hyperedgeData = this.dataGenerators.hyperedges.generate(size, baseNodes);
      const creationResults = await this.benchmarkOperation(
        `${implName}_hyperedge_creation_${size}`,
        async () => {
          const promises = hyperedgeData.map(data => kb.createHyperedge(data));
          return await Promise.all(promises);
        }
      );
      
      results.creation[size] = creationResults;
      
      // Hyperpattern matching
      if (kb.findHyperpattern) {
        const hyperpatternResults = await this.benchmarkOperation(
          `${implName}_hyperpattern_${size}`,
          async () => {
            return await kb.findHyperpattern({
              type: 'hyperedge_pattern',
              minSize: 3,
              maxSize: 10,
              limit: 100
            });
          }
        );
        
        results.hyperpatterns[size] = hyperpatternResults;
      }
    }
    
    return results;
  }

  /**
   * Benchmark batch operations
   */
  async benchmarkBatchOperations(kb, implName) {
    const results = {
      batchNodes: {},
      batchEdges: {},
      bulkImport: {}
    };
    
    for (const size of this.config.testSizes) {
      console.log(`    Testing batch operations with ${size} items...`);
      
      // Batch node creation
      if (kb.batchCreateNodes) {
        const nodeData = this.dataGenerators.nodes.generate(size);
        const batchNodeResults = await this.benchmarkOperation(
          `${implName}_batch_nodes_${size}`,
          async () => {
            return await kb.batchCreateNodes(nodeData);
          }
        );
        
        results.batchNodes[size] = batchNodeResults;
      }
      
      // Batch edge creation
      if (kb.batchCreateEdges && results.batchNodes[size]?.result) {
        const nodes = results.batchNodes[size].result;
        const edgeData = this.dataGenerators.edges.generate(
          Math.min(size, 10000), 
          nodes
        );
        
        const batchEdgeResults = await this.benchmarkOperation(
          `${implName}_batch_edges_${size}`,
          async () => {
            return await kb.batchCreateEdges(edgeData);
          }
        );
        
        results.batchEdges[size] = batchEdgeResults;
      }
    }
    
    return results;
  }

  /**
   * Benchmark concurrent operations
   */
  async benchmarkConcurrentOperations(kb, implName) {
    const results = {
      concurrentReads: {},
      concurrentWrites: {},
      mixed: {}
    };
    
    // Create base data
    const baseNodes = await this.createBaseNodes(kb, 10000);
    
    for (const concurrency of this.config.concurrencyLevels) {
      console.log(`    Testing with ${concurrency} concurrent operations...`);
      
      // Concurrent reads
      const readResults = await this.benchmarkOperation(
        `${implName}_concurrent_reads_${concurrency}`,
        async () => {
          const promises = Array.from({ length: concurrency }, () => 
            this.performConcurrentReads(kb, baseNodes, 100)
          );
          return await Promise.all(promises);
        }
      );
      
      results.concurrentReads[concurrency] = readResults;
      
      // Concurrent writes
      const writeResults = await this.benchmarkOperation(
        `${implName}_concurrent_writes_${concurrency}`,
        async () => {
          const promises = Array.from({ length: concurrency }, (_, i) => 
            this.performConcurrentWrites(kb, i * 100, 100)
          );
          return await Promise.all(promises);
        }
      );
      
      results.concurrentWrites[concurrency] = writeResults;
    }
    
    return results;
  }

  /**
   * Benchmark memory scalability
   */
  async benchmarkMemoryScalability(kb, implName) {
    const results = {
      memoryGrowth: [],
      cacheEfficiency: {},
      garbageCollection: {}
    };
    
    const sizes = [1000, 5000, 10000, 50000, 100000];
    
    for (const size of sizes) {
      console.log(`    Testing memory usage with ${size} nodes...`);
      
      const memoryBefore = process.memoryUsage();
      
      // Create nodes and measure memory
      await this.createBaseNodes(kb, size);
      
      const memoryAfter = process.memoryUsage();
      
      results.memoryGrowth.push({
        size,
        memoryBefore,
        memoryAfter,
        growth: memoryAfter.heapUsed - memoryBefore.heapUsed
      });
      
      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }
    }
    
    return results;
  }

  /**
   * Benchmark query performance
   */
  async benchmarkQueryPerformance(kb, implName) {
    const results = {
      simpleQueries: {},
      complexQueries: {},
      aggregations: {}
    };
    
    // Create comprehensive test data
    await this.createTestGraph(kb, 50000, 200000);
    
    // Simple queries
    const simpleQueryResults = await this.benchmarkOperation(
      `${implName}_simple_queries`,
      async () => {
        const queries = [
          () => kb.getNode ? kb.getNode('test_node_1') : null,
          () => kb.findNodes ? kb.findNodes({ type: 'Person' }) : null,
          () => kb.findEdges ? kb.findEdges({ type: 'KNOWS' }) : null
        ];
        
        const results = [];
        for (const query of queries) {
          if (query) {
            results.push(await query());
          }
        }
        return results;
      }
    );
    
    results.simpleQueries = simpleQueryResults;
    
    // Complex queries
    const complexQueryResults = await this.benchmarkOperation(
      `${implName}_complex_queries`,
      async () => {
        if (kb.findPattern) {
          return await kb.findPattern({
            nodes: {
              A: { type: 'Person' },
              B: { type: 'Person' },
              C: { type: 'Company' }
            },
            edges: [
              { from: 'A', to: 'B', type: 'KNOWS' },
              { from: 'A', to: 'C', type: 'WORKS_FOR' },
              { from: 'B', to: 'C', type: 'WORKS_FOR' }
            ]
          });
        }
        return null;
      }
    );
    
    results.complexQueries = complexQueryResults;
    
    return results;
  }

  /**
   * Benchmark index performance
   */
  async benchmarkIndexPerformance(kb, implName) {
    const results = {
      indexCreation: {},
      indexedQueries: {},
      indexMemory: {}
    };
    
    // Test different index scenarios
    const scenarios = [
      { name: 'type_index', field: 'type' },
      { name: 'property_index', field: 'properties' },
      { name: 'temporal_index', field: 'created' }
    ];
    
    for (const scenario of scenarios) {
      console.log(`    Testing ${scenario.name}...`);
      
      // Create index if supported
      if (kb.createIndex) {
        const indexResults = await this.benchmarkOperation(
          `${implName}_index_creation_${scenario.name}`,
          async () => {
            return await kb.createIndex(scenario.field);
          }
        );
        
        results.indexCreation[scenario.name] = indexResults;
      }
      
      // Test indexed queries
      const queryResults = await this.benchmarkOperation(
        `${implName}_indexed_query_${scenario.name}`,
        async () => {
          const filter = {};
          filter[scenario.field] = 'test_value';
          return kb.findNodes ? await kb.findNodes(filter) : null;
        }
      );
      
      results.indexedQueries[scenario.name] = queryResults;
    }
    
    return results;
  }

  /**
   * Core benchmarking operation
   */
  async benchmarkOperation(operationName, operation) {
    const iterations = this.config.iterations;
    const times = [];
    const memoryUsages = [];
    let result = null;
    let error = null;
    
    console.log(`      Running ${operationName}...`);
    
    for (let i = 0; i < iterations; i++) {
      const memoryBefore = process.memoryUsage();
      const startTime = performance.now();
      
      try {
        result = await Promise.race([
          operation(),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Operation timeout')), this.config.timeout)
          )
        ]);
        
        const endTime = performance.now();
        const memoryAfter = process.memoryUsage();
        
        times.push(endTime - startTime);
        memoryUsages.push({
          before: memoryBefore,
          after: memoryAfter,
          delta: memoryAfter.heapUsed - memoryBefore.heapUsed
        });
        
      } catch (err) {
        error = err.message;
        break;
      }
    }
    
    if (error) {
      return {
        operationName,
        error,
        timestamp: new Date().toISOString()
      };
    }
    
    return {
      operationName,
      iterations,
      times,
      memoryUsages,
      statistics: this.statistics.calculate(times),
      result: Array.isArray(result) ? result.length : (result ? 1 : 0),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Generate comprehensive benchmark report
   */
  async generateReport() {
    console.log('\n📊 Generating comprehensive benchmark report...');
    
    const report = {
      metadata: {
        timestamp: new Date().toISOString(),
        config: this.config,
        system: {
          platform: process.platform,
          arch: process.arch,
          nodeVersion: process.version,
          cpuCount: require('os').cpus().length,
          totalMemory: require('os').totalmem(),
          freeMemory: require('os').freemem()
        }
      },
      results: Object.fromEntries(this.results),
      summary: this.generateSummary(),
      recommendations: this.generateRecommendations()
    };
    
    // Save detailed results
    const reportPath = path.join(this.config.outputDir, 'benchmark-report.json');
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    
    // Generate HTML report
    await this.generateHtmlReport(report);
    
    // Generate CSV summaries
    await this.generateCsvReports(report);
    
    console.log('📄 Reports generated:');
    console.log(`  - JSON: ${reportPath}`);
    console.log(`  - HTML: ${path.join(this.config.outputDir, 'benchmark-report.html')}`);
    console.log(`  - CSV:  ${path.join(this.config.outputDir, 'benchmark-summary.csv')}`);
  }

  /**
   * Generate benchmark summary
   */
  generateSummary() {
    const summary = {};
    
    for (const [suiteName, suiteResults] of this.results) {
      summary[suiteName] = {};
      
      for (const [implName, implResults] of suiteResults) {
        if (implResults.error) {
          summary[suiteName][implName] = { error: implResults.error };
          continue;
        }
        
        // Calculate overall performance scores
        const scores = this.calculatePerformanceScores(implResults);
        summary[suiteName][implName] = scores;
      }
    }
    
    return summary;
  }

  /**
   * Calculate performance scores
   */
  calculatePerformanceScores(results) {
    let totalScore = 0;
    let testCount = 0;
    
    const traverse = (obj) => {
      for (const [key, value] of Object.entries(obj)) {
        if (value && typeof value === 'object' && value.statistics) {
          // Calculate score based on median time (lower is better)
          const medianTime = value.statistics.median;
          const score = Math.max(0, 100 - Math.log10(medianTime + 1) * 10);
          totalScore += score;
          testCount++;
        } else if (value && typeof value === 'object') {
          traverse(value);
        }
      }
    };
    
    traverse(results);
    
    return {
      overallScore: testCount > 0 ? totalScore / testCount : 0,
      testCount,
      details: results
    };
  }

  /**
   * Generate recommendations
   */
  generateRecommendations() {
    const recommendations = [];
    
    // Analyze results and generate recommendations
    for (const [suiteName, suiteResults] of this.results) {
      for (const [implName, implResults] of suiteResults) {
        if (implResults.error) {
          recommendations.push({
            implementation: implName,
            suite: suiteName,
            type: 'error',
            message: `Failed to complete ${suiteName}: ${implResults.error}`,
            priority: 'high'
          });
        } else {
          // Analyze performance patterns
          const memoryIssues = this.detectMemoryIssues(implResults);
          const performanceIssues = this.detectPerformanceIssues(implResults);
          
          recommendations.push(...memoryIssues, ...performanceIssues);
        }
      }
    }
    
    return recommendations;
  }

  /**
   * Detect memory issues
   */
  detectMemoryIssues(results) {
    const issues = [];
    
    // Traverse results and check for memory problems
    const traverse = (obj, path = '') => {
      for (const [key, value] of Object.entries(obj)) {
        if (value && value.memoryUsages) {
          const avgMemoryDelta = value.memoryUsages.reduce(
            (sum, usage) => sum + usage.delta, 0
          ) / value.memoryUsages.length;
          
          if (avgMemoryDelta > 100 * 1024 * 1024) { // 100MB
            issues.push({
              type: 'memory',
              message: `High memory usage detected in ${path}.${key}: ${Math.round(avgMemoryDelta / 1024 / 1024)}MB average`,
              priority: 'medium',
              path: `${path}.${key}`
            });
          }
        } else if (value && typeof value === 'object') {
          traverse(value, path ? `${path}.${key}` : key);
        }
      }
    };
    
    traverse(results);
    return issues;
  }

  /**
   * Detect performance issues
   */
  detectPerformanceIssues(results) {
    const issues = [];
    
    // Check for slow operations
    const traverse = (obj, path = '') => {
      for (const [key, value] of Object.entries(obj)) {
        if (value && value.statistics) {
          if (value.statistics.median > 10000) { // 10 seconds
            issues.push({
              type: 'performance',
              message: `Slow operation detected in ${path}.${key}: ${value.statistics.median.toFixed(2)}ms median`,
              priority: 'high',
              path: `${path}.${key}`
            });
          }
          
          if (value.statistics.max > value.statistics.median * 10) {
            issues.push({
              type: 'performance',
              message: `High variance detected in ${path}.${key}: max time is ${(value.statistics.max / value.statistics.median).toFixed(1)}x median`,
              priority: 'medium',
              path: `${path}.${key}`
            });
          }
        } else if (value && typeof value === 'object') {
          traverse(value, path ? `${path}.${key}` : key);
        }
      }
    };
    
    traverse(results);
    return issues;
  }

  /**
   * Generate HTML report
   */
  async generateHtmlReport(report) {
    const htmlTemplate = `
<!DOCTYPE html>
<html>
<head>
    <title>Knowledge Base Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .implementation { margin: 10px 0; padding: 10px; border: 1px solid #ddd; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
        .metric { padding: 10px; background: #f9f9f9; border-radius: 3px; }
        .error { background: #ffebee; color: #c62828; }
        .success { background: #e8f5e8; color: #2e7d32; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #f5f5f5; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Knowledge Base Benchmark Report</h1>
        <p>Generated: ${report.metadata.timestamp}</p>
        <p>System: ${report.metadata.system.platform} ${report.metadata.system.arch}</p>
    </div>
    
    ${this.generateHtmlSummary(report.summary)}
    ${this.generateHtmlRecommendations(report.recommendations)}
    ${this.generateHtmlDetails(report.results)}
</body>
</html>`;
    
    const htmlPath = path.join(this.config.outputDir, 'benchmark-report.html');
    await fs.writeFile(htmlPath, htmlTemplate);
  }

  /**
   * Helper methods for test data creation
   */
  async createBaseNodes(kb, count) {
    const nodes = [];
    const batchSize = 1000;
    
    for (let i = 0; i < count; i += batchSize) {
      const batch = Math.min(batchSize, count - i);
      const nodeData = this.dataGenerators.nodes.generate(batch);
      
      if (kb.batchCreateNodes) {
        const created = await kb.batchCreateNodes(nodeData);
        nodes.push(...created);
      } else {
        for (const data of nodeData) {
          const node = await kb.createNode(data);
          nodes.push(node);
        }
      }
    }
    
    return nodes;
  }

  async createTestGraph(kb, nodeCount, edgeCount) {
    const nodes = await this.createBaseNodes(kb, nodeCount);
    const edgeData = this.dataGenerators.edges.generate(edgeCount, nodes);
    
    if (kb.batchCreateEdges) {
      await kb.batchCreateEdges(edgeData);
    } else {
      for (const data of edgeData) {
        await kb.createEdge(data);
      }
    }
  }

  async performConcurrentReads(kb, nodes, count) {
    const promises = [];
    for (let i = 0; i < count; i++) {
      const randomNode = nodes[Math.floor(Math.random() * nodes.length)];
      promises.push(kb.getNode(randomNode.id));
    }
    return await Promise.all(promises);
  }

  async performConcurrentWrites(kb, startIndex, count) {
    const nodeData = this.dataGenerators.nodes.generate(count, startIndex);
    const promises = nodeData.map(data => kb.createNode(data));
    return await Promise.all(promises);
  }

  async runWarmup(kb, implName) {
    // Simple warmup operations
    const warmupNodes = this.dataGenerators.nodes.generate(100);
    
    for (const nodeData of warmupNodes) {
      try {
        await kb.createNode(nodeData);
      } catch (error) {
        // Ignore warmup errors
      }
    }
  }

  async cleanupImplementation(kb) {
    if (kb.clearCache) {
      kb.clearCache();
    }
    
    // Force garbage collection
    if (global.gc) {
      global.gc();
    }
  }

  async cleanup() {
    for (const [name, kb] of this.knowledgeBases) {
      try {
        if (kb.close) {
          await kb.close();
        } else if (kb.shutdown) {
          await kb.shutdown();
        }
      } catch (error) {
        console.warn(`Warning: Failed to cleanup ${name}:`, error.message);
      }
    }
  }
}

/**
 * Data Generators
 */
class NodeDataGenerator {
  generate(count, startIndex = 0) {
    const types = ['Person', 'Company', 'Project', 'Technology', 'Event'];
    const domains = ['tech', 'business', 'science', 'education', 'healthcare'];
    
    return Array.from({ length: count }, (_, i) => ({
      id: `benchmark_node_${startIndex + i}`,
      type: types[i % types.length],
      data: {
        name: `Node ${startIndex + i}`,
        value: Math.random() * 1000,
        category: `category_${i % 10}`,
        timestamp: Date.now(),
        metadata: {
          benchmarkGenerated: true,
          index: startIndex + i
        }
      },
      properties: {
        domain: domains[i % domains.length],
        active: Math.random() > 0.3,
        score: Math.random() * 100
      },
      labels: [`label_${i % 5}`, 'benchmark'],
      metadata: {
        confidence: 0.8 + Math.random() * 0.2,
        source: 'benchmark_generator'
      }
    }));
  }
}

class EdgeDataGenerator {
  generate(count, nodes) {
    const types = ['CONNECTS', 'RELATES_TO', 'DEPENDS_ON', 'INFLUENCES', 'CONTAINS'];
    
    return Array.from({ length: count }, (_, i) => {
      const fromNode = nodes[Math.floor(Math.random() * nodes.length)];
      const toNode = nodes[Math.floor(Math.random() * nodes.length)];
      
      return {
        id: `benchmark_edge_${i}`,
        type: types[i % types.length],
        from: fromNode.id,
        to: toNode.id,
        properties: {
          strength: Math.random(),
          weight: Math.random() * 10,
          created: Date.now()
        },
        metadata: {
          confidence: 0.7 + Math.random() * 0.3,
          source: 'benchmark_generator'
        },
        bidirectional: Math.random() > 0.7
      };
    });
  }
}

class PatternDataGenerator {
  generate(complexity = 'simple') {
    // Generate various pattern types for testing
    return {
      simple: {
        nodes: { A: { type: 'Person' }, B: { type: 'Company' } },
        edges: [{ from: 'A', to: 'B', type: 'WORKS_FOR' }]
      },
      complex: {
        nodes: {
          A: { type: 'Person' },
          B: { type: 'Company' }, 
          C: { type: 'Project' },
          D: { type: 'Technology' }
        },
        edges: [
          { from: 'A', to: 'B', type: 'WORKS_FOR' },
          { from: 'A', to: 'C', type: 'ASSIGNED_TO' },
          { from: 'C', to: 'D', type: 'USES' }
        ]
      }
    }[complexity];
  }
}

class HyperedgeDataGenerator {
  generate(count, nodes) {
    const types = ['COLLABORATION', 'MEETING', 'TRANSACTION', 'EVENT', 'GROUP'];
    
    return Array.from({ length: count }, (_, i) => {
      const nodeCount = 3 + Math.floor(Math.random() * 7); // 3-9 nodes per hyperedge
      const selectedNodes = [];
      
      for (let j = 0; j < nodeCount; j++) {
        const node = nodes[Math.floor(Math.random() * nodes.length)];
        if (!selectedNodes.includes(node.id)) {
          selectedNodes.push(node.id);
        }
      }
      
      return {
        id: `benchmark_hyperedge_${i}`,
        type: types[i % types.length],
        nodes: selectedNodes,
        properties: {
          strength: Math.random(),
          duration: Math.random() * 3600000, // up to 1 hour
          importance: Math.random() * 10
        },
        metadata: {
          confidence: 0.8 + Math.random() * 0.2,
          source: 'benchmark_generator'
        }
      };
    });
  }
}

/**
 * Performance Monitors
 */
class MemoryMonitor {
  constructor() {
    this.measurements = [];
    this.interval = null;
  }
  
  start() {
    this.measurements = [];
    this.interval = setInterval(() => {
      this.measurements.push({
        timestamp: Date.now(),
        memory: process.memoryUsage()
      });
    }, 1000);
  }
  
  stop() {
    if (this.interval) {
      clearInterval(this.interval);
    }
    
    return {
      measurements: this.measurements,
      peak: this.measurements.reduce((max, m) => 
        m.memory.heapUsed > max ? m.memory.heapUsed : max, 0),
      average: this.measurements.reduce((sum, m) => 
        sum + m.memory.heapUsed, 0) / this.measurements.length
    };
  }
}

class CpuMonitor {
  constructor() {
    this.startUsage = null;
  }
  
  start() {
    this.startUsage = process.cpuUsage();
  }
  
  stop() {
    const endUsage = process.cpuUsage(this.startUsage);
    return {
      user: endUsage.user,
      system: endUsage.system,
      total: endUsage.user + endUsage.system
    };
  }
}

class NetworkMonitor {
  // Placeholder for network monitoring
  constructor() {}
  start() {}
  stop() { return {}; }
}

/**
 * Statistics Calculator
 */
class StatisticsCollector {
  calculate(values) {
    if (!values || values.length === 0) {
      return { count: 0 };
    }
    
    const sorted = [...values].sort((a, b) => a - b);
    const sum = values.reduce((a, b) => a + b, 0);
    
    return {
      count: values.length,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      mean: sum / values.length,
      median: sorted[Math.floor(sorted.length / 2)],
      p95: sorted[Math.floor(sorted.length * 0.95)],
      p99: sorted[Math.floor(sorted.length * 0.99)],
      stdDev: this.calculateStdDev(values, sum / values.length)
    };
  }
  
  calculateStdDev(values, mean) {
    const variance = values.reduce((sum, value) => 
      sum + Math.pow(value - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }
}

module.exports = ComprehensiveBenchmark;

// Run benchmarks if executed directly
if (require.main === module) {
  const benchmark = new ComprehensiveBenchmark({
    implementations: process.env.BENCHMARK_IMPLEMENTATIONS?.split(',') || 
      ['neo4j', 'redis', 'hypergraph', 'custom'],
    testSizes: [1000, 10000, 100000],
    concurrencyLevels: [1, 10, 50],
    iterations: 3,
    outputDir: './benchmark-results'
  });
  
  benchmark.runBenchmarks().catch(error => {
    console.error('Benchmark failed:', error);
    process.exit(1);
  });
}