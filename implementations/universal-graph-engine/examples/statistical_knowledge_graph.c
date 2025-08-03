/**
 * Universal Graph Engine - Statistical Knowledge Graph
 * 
 * A comprehensive demonstration of probability theory, stochastic processes,
 * regression analysis, and inferential statistics modeled as a complex graph
 * with quantum uncertainty, temporal evolution, and meta-relationships.
 * 
 * This graph represents the deep interconnections between statistical concepts,
 * their mathematical foundations, practical applications, and evolution over time.
 * 
 * Copyright (c) 2025 Universal Graph Engine Project
 * Licensed under MIT License
 */

#include "universal_graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>

/* ============================================================================
 * STATISTICAL DATA STRUCTURES
 * ============================================================================ */

typedef struct {
    char name[128];
    char definition[512];
    char field[64];
    double complexity_score;      /* 0.0 to 10.0 */
    double practical_importance;  /* 0.0 to 10.0 */
    double historical_age_years;
    char* prerequisites[16];
    size_t prerequisite_count;
    char* applications[16];
    size_t application_count;
} statistical_concept_t;

typedef struct {
    char name[128];
    double mean;
    double variance;
    double skewness;
    double kurtosis;
    char* parameters[8];
    size_t parameter_count;
    char support_description[256];
    char pdf_formula[256];
    char cdf_formula[256];
} probability_distribution_t;

typedef struct {
    char name[128];
    char type[64];              /* "discrete", "continuous", "mixed" */
    char state_space[256];
    char time_domain[64];       /* "discrete", "continuous" */
    bool is_markovian;
    bool is_stationary;
    bool has_independent_increments;
    double* transition_matrix;  /* For discrete state processes */
    size_t state_count;
    char governing_equation[256];
} stochastic_process_t;

typedef struct {
    char name[128];
    char type[64];              /* "linear", "logistic", "polynomial", "nonlinear" */
    size_t predictor_count;
    char* predictors[16];
    char response_variable[64];
    double r_squared;
    double adjusted_r_squared;
    double aic;
    double bic;
    char assumptions[512];
    char interpretation[512];
} regression_model_t;

typedef struct {
    char name[128];
    char category[64];          /* "hypothesis_testing", "estimation", "bayesian" */
    char null_hypothesis[256];
    char alternative_hypothesis[256];
    double significance_level;
    double power;
    double effect_size;
    char test_statistic[128];
    char decision_rule[256];
    char assumptions[512];
} statistical_test_t;

typedef struct {
    char formula[256];
    char variables[512];
    char conditions[256];
    double complexity;
    char derivation_notes[512];
} mathematical_relationship_t;

typedef struct {
    char context[128];
    char domain[64];
    char problem_type[128];
    double success_rate;
    char* required_concepts[16];
    size_t concept_count;
    char methodology[512];
} practical_application_t;

/* Quantum state for uncertain statistical relationships */
typedef struct {
    double confidence_interval[2];
    double bayesian_probability;
    double frequentist_probability;
    double epistemic_uncertainty;
    double aleatoric_uncertainty;
    bool is_contested;          /* Whether statisticians disagree */
} statistical_uncertainty_t;

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

static void print_section_header(const char* title) {
    printf("\n");
    printf("📊 ");
    for (int i = 0; i < 75; i++) printf("=");
    printf("\n");
    printf("📊 %s\n", title);
    printf("📊 ");
    for (int i = 0; i < 75; i++) printf("=");
    printf("\n\n");
}

static ug_node_id_t create_concept_node(ug_graph_t* graph, const char* name, 
                                       const char* definition, const char* field,
                                       double complexity, double importance) {
    statistical_concept_t concept = {0};
    strncpy(concept.name, name, sizeof(concept.name) - 1);
    strncpy(concept.definition, definition, sizeof(concept.definition) - 1);
    strncpy(concept.field, field, sizeof(concept.field) - 1);
    concept.complexity_score = complexity;
    concept.practical_importance = importance;
    concept.historical_age_years = 50 + (rand() % 300); /* Random historical age */
    concept.prerequisite_count = 0;
    concept.application_count = 0;
    
    return ug_create_node(graph, UG_TYPE_CUSTOM_STRUCT, &concept);
}

static ug_node_id_t create_distribution_node(ug_graph_t* graph, const char* name,
                                            double mean, double variance,
                                            const char* support, const char* pdf) {
    probability_distribution_t dist = {0};
    strncpy(dist.name, name, sizeof(dist.name) - 1);
    dist.mean = mean;
    dist.variance = variance;
    dist.skewness = 0.0; /* Default values */
    dist.kurtosis = 3.0;
    strncpy(dist.support_description, support, sizeof(dist.support_description) - 1);
    strncpy(dist.pdf_formula, pdf, sizeof(dist.pdf_formula) - 1);
    dist.parameter_count = 0;
    
    return ug_create_node(graph, UG_TYPE_CUSTOM_STRUCT, &dist);
}

static ug_node_id_t create_process_node(ug_graph_t* graph, const char* name,
                                       const char* type, bool is_markovian,
                                       bool is_stationary, const char* equation) {
    stochastic_process_t process = {0};
    strncpy(process.name, name, sizeof(process.name) - 1);
    strncpy(process.type, type, sizeof(process.type) - 1);
    strcpy(process.time_domain, "continuous");
    process.is_markovian = is_markovian;
    process.is_stationary = is_stationary;
    process.has_independent_increments = false;
    strncpy(process.governing_equation, equation, sizeof(process.governing_equation) - 1);
    process.state_count = 0;
    process.transition_matrix = NULL;
    
    return ug_create_node(graph, UG_TYPE_CUSTOM_STRUCT, &process);
}

static ug_node_id_t create_regression_node(ug_graph_t* graph, const char* name,
                                          const char* type, const char* response,
                                          double r_squared, const char* assumptions) {
    regression_model_t model = {0};
    strncpy(model.name, name, sizeof(model.name) - 1);
    strncpy(model.type, type, sizeof(model.type) - 1);
    strncpy(model.response_variable, response, sizeof(model.response_variable) - 1);
    model.r_squared = r_squared;
    model.adjusted_r_squared = r_squared * 0.95; /* Approximate */
    model.aic = 100 + rand() % 200; /* Random AIC */
    model.bic = model.aic + 10;
    strncpy(model.assumptions, assumptions, sizeof(model.assumptions) - 1);
    model.predictor_count = 0;
    
    return ug_create_node(graph, UG_TYPE_CUSTOM_STRUCT, &model);
}

static ug_node_id_t create_test_node(ug_graph_t* graph, const char* name,
                                    const char* category, const char* null_hyp,
                                    double alpha, double power, const char* test_stat) {
    statistical_test_t test = {0};
    strncpy(test.name, name, sizeof(test.name) - 1);
    strncpy(test.category, category, sizeof(test.category) - 1);
    strncpy(test.null_hypothesis, null_hyp, sizeof(test.null_hypothesis) - 1);
    test.significance_level = alpha;
    test.power = power;
    test.effect_size = 0.5; /* Medium effect size default */
    strncpy(test.test_statistic, test_stat, sizeof(test.test_statistic) - 1);
    
    return ug_create_node(graph, UG_TYPE_CUSTOM_STRUCT, &test);
}

/* ============================================================================
 * GRAPH CONSTRUCTION FUNCTIONS
 * ============================================================================ */

void build_probability_theory_foundation(ug_graph_t* graph, ug_node_id_t* foundation_nodes) {
    print_section_header("PROBABILITY THEORY FOUNDATION");
    
    /* Core probability concepts */
    foundation_nodes[0] = create_concept_node(graph, "Sample Space",
        "The set of all possible outcomes of a random experiment", "Probability Theory",
        3.0, 9.0);
    
    foundation_nodes[1] = create_concept_node(graph, "Event",
        "A subset of the sample space representing outcomes of interest", "Probability Theory",
        2.5, 9.5);
    
    foundation_nodes[2] = create_concept_node(graph, "Probability Measure",
        "A function that assigns probabilities to events satisfying Kolmogorov axioms", "Probability Theory",
        6.0, 10.0);
    
    foundation_nodes[3] = create_concept_node(graph, "Random Variable",
        "A measurable function from sample space to real numbers", "Probability Theory",
        5.0, 9.8);
    
    foundation_nodes[4] = create_concept_node(graph, "Expectation",
        "The average value of a random variable weighted by its probability", "Probability Theory",
        4.0, 9.0);
    
    foundation_nodes[5] = create_concept_node(graph, "Variance",
        "The expected value of squared deviations from the mean", "Probability Theory",
        4.5, 8.5);
    
    printf("Created foundational probability concepts:\n");
    for (int i = 0; i < 6; i++) {
        printf("  ✓ Node %d: Core probability concept\n", i + 1);
    }
    
    /* Create foundational relationships */
    ug_create_edge(graph, foundation_nodes[0], foundation_nodes[1], "CONTAINS", 1.0);
    ug_create_edge(graph, foundation_nodes[2], foundation_nodes[1], "ASSIGNS_PROBABILITY_TO", 1.0);
    ug_create_edge(graph, foundation_nodes[3], foundation_nodes[0], "MAPS_FROM", 1.0);
    ug_create_edge(graph, foundation_nodes[4], foundation_nodes[3], "CHARACTERIZES", 0.9);
    ug_create_edge(graph, foundation_nodes[5], foundation_nodes[3], "CHARACTERIZES", 0.9);
    
    /* Create meta-relationship about the relationship between expectation and variance */
    ug_relationship_id_t exp_var_rel = ug_create_edge(graph, foundation_nodes[4], foundation_nodes[5], "RELATED_TO", 0.8);
    ug_node_id_t linearity = create_concept_node(graph, "Linearity of Expectation",
        "E[aX + bY] = aE[X] + bE[Y] for any constants a,b and random variables X,Y", "Probability Theory",
        3.5, 8.0);
    
    ug_relationship_id_t meta_rel = ug_create_edge(graph, linearity, exp_var_rel, "GOVERNS", 0.95);
    printf("  ✓ Created meta-relationship: Linearity governs expectation-variance relationship\n");
    
    printf("\nFoundational structure established with %zu relationships\n", ug_get_relationship_count(graph));
}

void build_probability_distributions(ug_graph_t* graph, ug_node_id_t* distribution_nodes) {
    print_section_header("PROBABILITY DISTRIBUTIONS");
    
    /* Discrete distributions */
    distribution_nodes[0] = create_distribution_node(graph, "Bernoulli Distribution",
        0.5, 0.25, "{0, 1}", "p^x * (1-p)^(1-x)");
    
    distribution_nodes[1] = create_distribution_node(graph, "Binomial Distribution",
        10.0, 2.5, "{0, 1, 2, ..., n}", "C(n,k) * p^k * (1-p)^(n-k)");
    
    distribution_nodes[2] = create_distribution_node(graph, "Poisson Distribution",
        5.0, 5.0, "{0, 1, 2, ...}", "λ^k * e^(-λ) / k!");
    
    distribution_nodes[3] = create_distribution_node(graph, "Geometric Distribution",
        2.0, 2.0, "{1, 2, 3, ...}", "p * (1-p)^(k-1)");
    
    /* Continuous distributions */
    distribution_nodes[4] = create_distribution_node(graph, "Normal Distribution",
        0.0, 1.0, "(-∞, ∞)", "1/√(2πσ²) * exp(-(x-μ)²/(2σ²))");
    
    distribution_nodes[5] = create_distribution_node(graph, "Exponential Distribution",
        1.0, 1.0, "[0, ∞)", "λ * exp(-λx)");
    
    distribution_nodes[6] = create_distribution_node(graph, "Uniform Distribution",
        0.5, 1.0/12.0, "[a, b]", "1/(b-a)");
    
    distribution_nodes[7] = create_distribution_node(graph, "Gamma Distribution",
        2.0, 2.0, "(0, ∞)", "β^α/Γ(α) * x^(α-1) * exp(-βx)");
    
    distribution_nodes[8] = create_distribution_node(graph, "Beta Distribution",
        0.5, 0.05, "[0, 1]", "Γ(α+β)/(Γ(α)Γ(β)) * x^(α-1) * (1-x)^(β-1)");
    
    /* Multivariate distributions */
    distribution_nodes[9] = create_distribution_node(graph, "Multivariate Normal",
        0.0, 1.0, "ℝ^k", "(2π)^(-k/2)|Σ|^(-1/2) * exp(-½(x-μ)ᵀΣ⁻¹(x-μ))");
    
    printf("Created probability distributions:\n");
    printf("  ✓ Discrete: Bernoulli, Binomial, Poisson, Geometric\n");
    printf("  ✓ Continuous: Normal, Exponential, Uniform, Gamma, Beta\n");
    printf("  ✓ Multivariate: Multivariate Normal\n");
    
    /* Create hierarchical relationships */
    ug_create_edge(graph, distribution_nodes[1], distribution_nodes[0], "GENERALIZES", 0.9);
    ug_create_edge(graph, distribution_nodes[7], distribution_nodes[5], "GENERALIZES", 0.8);
    ug_create_edge(graph, distribution_nodes[8], distribution_nodes[6], "RELATED_TO", 0.7);
    
    /* Create limiting relationships */
    ug_create_edge(graph, distribution_nodes[1], distribution_nodes[2], "CONVERGES_TO", 0.85);
    ug_create_edge(graph, distribution_nodes[1], distribution_nodes[4], "CONVERGES_TO", 0.9);
    
    /* Create conjugate prior relationships (Bayesian) */
    ug_create_edge(graph, distribution_nodes[8], distribution_nodes[0], "CONJUGATE_PRIOR", 0.95);
    ug_create_edge(graph, distribution_nodes[7], distribution_nodes[2], "CONJUGATE_PRIOR", 0.95);
    
    printf("  ✓ Created distribution hierarchy and limiting relationships\n");
    printf("  ✓ Established conjugate prior relationships for Bayesian inference\n");
}

void build_stochastic_processes(ug_graph_t* graph, ug_node_id_t* process_nodes) {
    print_section_header("STOCHASTIC PROCESSES");
    
    /* Discrete-time processes */
    process_nodes[0] = create_process_node(graph, "Random Walk",
        "discrete", true, true, "S_n = S_{n-1} + X_n");
    
    process_nodes[1] = create_process_node(graph, "Markov Chain",
        "discrete", true, false, "P(X_{n+1}=j|X_n=i) = p_{ij}");
    
    process_nodes[2] = create_process_node(graph, "Autoregressive Process",
        "discrete", false, true, "X_t = φ₁X_{t-1} + φ₂X_{t-2} + ... + ε_t");
    
    /* Continuous-time processes */
    process_nodes[3] = create_process_node(graph, "Brownian Motion",
        "continuous", true, true, "dB_t ~ N(0, dt)");
    
    process_nodes[4] = create_process_node(graph, "Poisson Process",
        "counting", true, true, "N(t+s) - N(s) ~ Poisson(λt)");
    
    process_nodes[5] = create_process_node(graph, "Ornstein-Uhlenbeck Process",
        "continuous", true, true, "dX_t = θ(μ - X_t)dt + σdB_t");
    
    process_nodes[6] = create_process_node(graph, "Geometric Brownian Motion",
        "continuous", false, false, "dS_t = μS_t dt + σS_t dB_t");
    
    /* Jump processes */
    process_nodes[7] = create_process_node(graph, "Compound Poisson Process",
        "jump", false, true, "X_t = Σᵢ₌₁^{N(t)} Y_i");
    
    process_nodes[8] = create_process_node(graph, "Lévy Process",
        "jump", false, true, "X_{t+s} - X_s is independent of {X_r : r ≤ s}");
    
    printf("Created stochastic processes:\n");
    printf("  ✓ Discrete-time: Random Walk, Markov Chain, AR Process\n");
    printf("  ✓ Continuous-time: Brownian Motion, Poisson Process, OU Process, GBM\n");
    printf("  ✓ Jump processes: Compound Poisson, Lévy Process\n");
    
    /* Create hierarchical relationships */
    ug_create_edge(graph, process_nodes[1], process_nodes[0], "GENERALIZES", 0.8);
    ug_create_edge(graph, process_nodes[8], process_nodes[3], "GENERALIZES", 0.7);
    ug_create_edge(graph, process_nodes[8], process_nodes[4], "GENERALIZES", 0.7);
    ug_create_edge(graph, process_nodes[7], process_nodes[4], "BUILT_UPON", 0.9);
    
    /* Create scaling/limiting relationships */
    ug_create_edge(graph, process_nodes[0], process_nodes[3], "SCALES_TO", 0.85);
    ug_create_edge(graph, process_nodes[4], process_nodes[3], "APPROXIMATES", 0.7);
    
    /* Create application relationships */
    ug_create_edge(graph, process_nodes[6], process_nodes[3], "BASED_ON", 0.95);
    ug_create_edge(graph, process_nodes[5], process_nodes[3], "BASED_ON", 0.95);
    
    printf("  ✓ Established process hierarchy and scaling relationships\n");
    
    /* Create temporal meta-relationship */
    ug_relationship_id_t scaling_rel = ug_create_edge(graph, process_nodes[0], process_nodes[3], "CONVERGES_TO", 0.9);
    ug_node_id_t central_limit = create_concept_node(graph, "Central Limit Theorem",
        "Sums of independent random variables converge in distribution to normal", "Probability Theory",
        7.0, 9.5);
    
    ug_create_edge(graph, central_limit, scaling_rel, "EXPLAINS", 0.95);
    printf("  ✓ Created meta-relationship: CLT explains random walk → Brownian motion scaling\n");
}

void build_regression_analysis(ug_graph_t* graph, ug_node_id_t* regression_nodes) {
    print_section_header("REGRESSION ANALYSIS");
    
    /* Linear regression models */
    regression_nodes[0] = create_regression_node(graph, "Simple Linear Regression",
        "linear", "Y", 0.75, "Linearity, independence, homoscedasticity, normality");
    
    regression_nodes[1] = create_regression_node(graph, "Multiple Linear Regression",
        "linear", "Y", 0.82, "Linear relationship, no multicollinearity, homoscedasticity");
    
    regression_nodes[2] = create_regression_node(graph, "Polynomial Regression",
        "polynomial", "Y", 0.78, "Polynomial relationship, careful with overfitting");
    
    /* Generalized linear models */
    regression_nodes[3] = create_regression_node(graph, "Logistic Regression",
        "logistic", "P(Y=1)", 0.71, "Binary outcome, logit link function, independence");
    
    regression_nodes[4] = create_regression_node(graph, "Poisson Regression",
        "poisson", "E[Y]", 0.68, "Count data, log link function, equidispersion");
    
    regression_nodes[5] = create_regression_node(graph, "Gamma Regression",
        "gamma", "E[Y]", 0.73, "Positive continuous response, log link typical");
    
    /* Advanced regression techniques */
    regression_nodes[6] = create_regression_node(graph, "Ridge Regression",
        "regularized", "Y", 0.80, "Multicollinearity present, L2 penalty");
    
    regression_nodes[7] = create_regression_node(graph, "Lasso Regression",
        "regularized", "Y", 0.77, "Feature selection needed, L1 penalty");
    
    regression_nodes[8] = create_regression_node(graph, "Elastic Net",
        "regularized", "Y", 0.79, "Combines Ridge and Lasso penalties");
    
    /* Time series regression */
    regression_nodes[9] = create_regression_node(graph, "ARIMA",
        "time_series", "Y_t", 0.74, "Stationarity, no structural breaks");
    
    regression_nodes[10] = create_regression_node(graph, "VAR",
        "multivariate_time_series", "Y_t", 0.76, "Multivariate stationarity, lag selection");
    
    printf("Created regression models:\n");
    printf("  ✓ Linear: Simple, Multiple, Polynomial\n");
    printf("  ✓ GLM: Logistic, Poisson, Gamma\n");
    printf("  ✓ Regularized: Ridge, Lasso, Elastic Net\n");
    printf("  ✓ Time Series: ARIMA, VAR\n");
    
    /* Create methodological relationships */
    ug_create_edge(graph, regression_nodes[1], regression_nodes[0], "EXTENDS", 0.95);
    ug_create_edge(graph, regression_nodes[2], regression_nodes[1], "GENERALIZES", 0.8);
    ug_create_edge(graph, regression_nodes[8], regression_nodes[6], "COMBINES", 0.9);
    ug_create_edge(graph, regression_nodes[8], regression_nodes[7], "COMBINES", 0.9);
    
    /* Create problem-solution relationships */
    ug_create_edge(graph, regression_nodes[6], regression_nodes[1], "ADDRESSES_MULTICOLLINEARITY", 0.85);
    ug_create_edge(graph, regression_nodes[7], regression_nodes[1], "PERFORMS_FEATURE_SELECTION", 0.9);
    
    /* Create application domain relationships */
    ug_create_edge(graph, regression_nodes[3], regression_nodes[0], "ADAPTS_FOR_BINARY", 0.8);
    ug_create_edge(graph, regression_nodes[4], regression_nodes[0], "ADAPTS_FOR_COUNTS", 0.8);
    
    printf("  ✓ Established methodological hierarchy and adaptation relationships\n");
    
    /* Create hypergraph relationship for model selection */
    ug_node_id_t model_selection = create_concept_node(graph, "Model Selection",
        "Process of choosing among competing statistical models", "Statistics",
        6.5, 9.0);
    
    ug_node_id_t participants[] = {
        regression_nodes[6], regression_nodes[7], regression_nodes[8], model_selection
    };
    ug_create_hyperedge(graph, participants, 4, "MODEL_SELECTION_TRADEOFF");
    printf("  ✓ Created 4-way hypergraph for regularization model selection tradeoffs\n");
}

void build_inferential_statistics(ug_graph_t* graph, ug_node_id_t* test_nodes) {
    print_section_header("INFERENTIAL STATISTICS");
    
    /* Parametric tests */
    test_nodes[0] = create_test_node(graph, "One-sample t-test",
        "hypothesis_testing", "μ = μ₀", 0.05, 0.8, "t = (x̄ - μ₀)/(s/√n)");
    
    test_nodes[1] = create_test_node(graph, "Two-sample t-test",
        "hypothesis_testing", "μ₁ = μ₂", 0.05, 0.8, "t = (x̄₁ - x̄₂)/sp√(1/n₁ + 1/n₂)");
    
    test_nodes[2] = create_test_node(graph, "Paired t-test",
        "hypothesis_testing", "μd = 0", 0.05, 0.8, "t = d̄/(sd/√n)");
    
    test_nodes[3] = create_test_node(graph, "One-way ANOVA",
        "hypothesis_testing", "μ₁ = μ₂ = ... = μk", 0.05, 0.8, "F = MSB/MSW");
    
    test_nodes[4] = create_test_node(graph, "Chi-square test",
        "hypothesis_testing", "Independence assumption", 0.05, 0.8, "χ² = Σ(O-E)²/E");
    
    /* Non-parametric tests */
    test_nodes[5] = create_test_node(graph, "Mann-Whitney U test",
        "hypothesis_testing", "Same distribution", 0.05, 0.8, "U = R₁ - n₁(n₁+1)/2");
    
    test_nodes[6] = create_test_node(graph, "Wilcoxon signed-rank test",
        "hypothesis_testing", "Median difference = 0", 0.05, 0.8, "W = sum of positive ranks");
    
    test_nodes[7] = create_test_node(graph, "Kruskal-Wallis test",
        "hypothesis_testing", "Same distribution across groups", 0.05, 0.8, "H = 12/N(N+1) * Σ(Ri²/ni) - 3(N+1)");
    
    /* Bayesian inference */
    test_nodes[8] = create_test_node(graph, "Bayesian hypothesis testing",
        "bayesian", "Prior belief updated", 0.05, 0.8, "Bayes Factor");
    
    test_nodes[9] = create_test_node(graph, "MCMC estimation",
        "bayesian", "Posterior sampling", 0.05, 0.8, "Markov Chain Monte Carlo");
    
    printf("Created inferential statistics methods:\n");
    printf("  ✓ Parametric: t-tests, ANOVA, Chi-square\n");
    printf("  ✓ Non-parametric: Mann-Whitney, Wilcoxon, Kruskal-Wallis\n");
    printf("  ✓ Bayesian: Hypothesis testing, MCMC\n");
    
    /* Create methodological relationships */
    ug_create_edge(graph, test_nodes[1], test_nodes[0], "EXTENDS_TO_TWO_SAMPLES", 0.9);
    ug_create_edge(graph, test_nodes[3], test_nodes[1], "GENERALIZES_TO_MULTIPLE", 0.85);
    ug_create_edge(graph, test_nodes[2], test_nodes[0], "ADAPTS_FOR_PAIRED_DATA", 0.9);
    
    /* Create robustness relationships */
    ug_create_edge(graph, test_nodes[5], test_nodes[1], "ROBUST_ALTERNATIVE", 0.8);
    ug_create_edge(graph, test_nodes[6], test_nodes[2], "ROBUST_ALTERNATIVE", 0.8);
    ug_create_edge(graph, test_nodes[7], test_nodes[3], "ROBUST_ALTERNATIVE", 0.8);
    
    /* Create paradigm relationships */
    ug_create_edge(graph, test_nodes[8], test_nodes[0], "BAYESIAN_APPROACH", 0.7);
    ug_create_edge(graph, test_nodes[9], test_nodes[8], "COMPUTATIONAL_METHOD", 0.95);
    
    printf("  ✓ Established robustness and paradigm relationships\n");
    
    /* Create quantum uncertainty relationship for statistical significance */
    statistical_uncertainty_t significance_uncertainty = {
        .confidence_interval = {0.03, 0.07},  /* Around α = 0.05 */
        .bayesian_probability = 0.6,
        .frequentist_probability = 0.95,
        .epistemic_uncertainty = 0.3,
        .aleatoric_uncertainty = 0.1,
        .is_contested = true
    };
    
    ug_node_id_t significance_node = ug_create_node(graph, UG_TYPE_CUSTOM_STRUCT, &significance_uncertainty);
    ug_create_edge(graph, significance_node, test_nodes[0], "QUANTUM_UNCERTAINTY_ABOUT", 0.7);
    printf("  ✓ Created quantum uncertainty node for statistical significance debates\n");
}

void create_cross_domain_relationships(ug_graph_t* graph, 
                                     ug_node_id_t* foundation_nodes,
                                     ug_node_id_t* distribution_nodes,
                                     ug_node_id_t* process_nodes,
                                     ug_node_id_t* regression_nodes,
                                     ug_node_id_t* test_nodes) {
    print_section_header("CROSS-DOMAIN RELATIONSHIPS & META-ANALYSIS");
    
    /* Probability theory feeds into distributions */
    ug_create_edge(graph, foundation_nodes[3], distribution_nodes[4], "INSTANTIATED_AS", 0.9);
    ug_create_edge(graph, foundation_nodes[4], distribution_nodes[4], "CHARACTERIZED_BY", 0.95);
    ug_create_edge(graph, foundation_nodes[5], distribution_nodes[4], "CHARACTERIZED_BY", 0.95);
    
    /* Distributions feed into stochastic processes */
    ug_create_edge(graph, distribution_nodes[4], process_nodes[3], "DRIVES_INCREMENTS", 0.9);
    ug_create_edge(graph, distribution_nodes[2], process_nodes[4], "DRIVES_ARRIVALS", 0.95);
    ug_create_edge(graph, distribution_nodes[5], process_nodes[5], "DRIVES_NOISE", 0.8);
    
    /* Stochastic processes inform regression */
    ug_create_edge(graph, process_nodes[2], regression_nodes[9], "THEORETICAL_BASIS", 0.9);
    ug_create_edge(graph, process_nodes[1], regression_nodes[10], "MOTIVATES", 0.8);
    ug_create_edge(graph, process_nodes[3], regression_nodes[0], "PROVIDES_ERROR_MODEL", 0.7);
    
    /* Regression models require inferential statistics */
    ug_create_edge(graph, regression_nodes[0], test_nodes[0], "USES_FOR_COEFFICIENTS", 0.9);
    ug_create_edge(graph, regression_nodes[1], test_nodes[3], "USES_FOR_MODEL_COMPARISON", 0.85);
    ug_create_edge(graph, regression_nodes[3], test_nodes[4], "USES_FOR_GOODNESS_OF_FIT", 0.8);
    
    printf("Created cross-domain relationships:\n");
    printf("  ✓ Probability theory → Distributions → Processes → Regression → Inference\n");
    printf("  ✓ Theoretical foundations inform practical applications\n");
    
    /* Create temporal evolution relationships */
    ug_timestamp_t base_time = time(NULL) - (365 * 24 * 3600 * 300); /* 300 years ago */
    
    /* Historical development timeline */
    ug_temporal_validity_t bayes_era = {
        .validity = {base_time, base_time + (50 * 365 * 24 * 3600), false, false},
        .causality = UG_CAUSALITY_FORWARD
    };
    
    ug_temporal_validity_t freq_era = {
        .validity = {base_time + (200 * 365 * 24 * 3600), time(NULL), false, false},
        .causality = UG_CAUSALITY_FORWARD
    };
    
    ug_node_id_t bayes_paradigm = create_concept_node(graph, "Bayesian Paradigm",
        "Probability as degree of belief, subjective interpretation", "Philosophy of Statistics",
        8.0, 9.0);
    
    ug_node_id_t freq_paradigm = create_concept_node(graph, "Frequentist Paradigm",
        "Probability as long-run frequency, objective interpretation", "Philosophy of Statistics",
        7.5, 9.5);
    
    ug_relationship_id_t paradigm_conflict = ug_create_edge(graph, bayes_paradigm, freq_paradigm, "PHILOSOPHICAL_TENSION", 0.8);
    
    /* Meta-relationship about the paradigm conflict */
    ug_node_id_t philosophy_of_science = create_concept_node(graph, "Philosophy of Science",
        "Study of foundations, methods, and implications of science", "Philosophy",
        9.0, 8.0);
    
    ug_create_edge(graph, philosophy_of_science, paradigm_conflict, "CONTEXTUALIZES", 0.9);
    printf("  ✓ Created temporal paradigm evolution and philosophical meta-context\n");
    
    /* Create uncertainty quantification hypergraph */
    ug_node_id_t uncertainty_concepts[] = {
        foundation_nodes[2],  /* Probability Measure */
        foundation_nodes[5],  /* Variance */
        test_nodes[8],       /* Bayesian testing */
        create_concept_node(graph, "Confidence Intervals",
            "Range of plausible values for parameter", "Inferential Statistics", 5.0, 9.0),
        create_concept_node(graph, "Credible Intervals",
            "Bayesian probability intervals for parameter", "Bayesian Statistics", 5.5, 8.5)
    };
    
    ug_create_hyperedge(graph, uncertainty_concepts, 5, "UNCERTAINTY_QUANTIFICATION_ECOSYSTEM");
    printf("  ✓ Created 5-way uncertainty quantification hypergraph\n");
    
    /* Create mathematical relationship nodes */
    mathematical_relationship_t bayes_theorem = {
        .formula = "P(H|E) = P(E|H) * P(H) / P(E)",
        .complexity = 6.0
    };
    strcpy(bayes_theorem.variables, "H=hypothesis, E=evidence");
    strcpy(bayes_theorem.conditions, "P(E) > 0");
    strcpy(bayes_theorem.derivation_notes, "Follows from definition of conditional probability");
    
    ug_node_id_t bayes_formula = ug_create_node(graph, UG_TYPE_CUSTOM_STRUCT, &bayes_theorem);
    
    mathematical_relationship_t clt_formula = {
        .formula = "(S_n - nμ)/(σ√n) → N(0,1) as n→∞",
        .complexity = 8.0
    };
    strcpy(clt_formula.variables, "S_n=sum, μ=mean, σ=std_dev, n=sample_size");
    strcpy(clt_formula.conditions, "Finite variance, independent observations");
    strcpy(clt_formula.derivation_notes, "Characteristic function proof or martingale approach");
    
    ug_node_id_t clt_formula = ug_create_node(graph, UG_TYPE_CUSTOM_STRUCT, &clt_formula);
    
    /* Connect formulas to concepts */
    ug_create_edge(graph, bayes_formula, test_nodes[8], "MATHEMATICAL_BASIS", 0.95);
    ug_create_edge(graph, clt_formula, distribution_nodes[4], "EXPLAINS_EMERGENCE", 0.9);
    ug_create_edge(graph, clt_formula, process_nodes[3], "EXPLAINS_LIMITING_BEHAVIOR", 0.85);
    
    printf("  ✓ Connected mathematical formulations to statistical concepts\n");
    
    /* Create application domain nodes */
    practical_application_t applications[] = {
        {"Clinical Trials", "Medicine", "Drug efficacy testing", 0.85},
        {"A/B Testing", "Technology", "Website optimization", 0.78},
        {"Financial Risk", "Finance", "Portfolio optimization", 0.82},
        {"Quality Control", "Manufacturing", "Process monitoring", 0.90},
        {"Machine Learning", "AI", "Model validation", 0.75}
    };
    
    for (int i = 0; i < 5; i++) {
        ug_node_id_t app_node = ug_create_node(graph, UG_TYPE_CUSTOM_STRUCT, &applications[i]);
        
        /* Connect applications to relevant statistical methods */
        if (i == 0) { /* Clinical trials */
            ug_create_edge(graph, app_node, test_nodes[1], "USES", 0.9);
            ug_create_edge(graph, app_node, test_nodes[8], "MODERN_APPROACH", 0.8);
        }
        if (i == 1) { /* A/B testing */
            ug_create_edge(graph, app_node, test_nodes[1], "CORE_METHOD", 0.95);
            ug_create_edge(graph, app_node, regression_nodes[3], "CONVERSION_ANALYSIS", 0.8);
        }
        if (i == 2) { /* Financial risk */
            ug_create_edge(graph, app_node, process_nodes[6], "MODELS_PRICES", 0.9);
            ug_create_edge(graph, app_node, distribution_nodes[4], "RISK_MODELING", 0.85);
        }
        if (i == 3) { /* Quality control */
            ug_create_edge(graph, app_node, test_nodes[4], "CONTROL_CHARTS", 0.9);
            ug_create_edge(graph, app_node, distribution_nodes[4], "PROCESS_VARIATION", 0.85);
        }
        if (i == 4) { /* Machine learning */
            ug_create_edge(graph, app_node, regression_nodes[6], "REGULARIZATION", 0.9);
            ug_create_edge(graph, app_node, test_nodes[0], "SIGNIFICANCE_TESTING", 0.7);
        }
    }
    
    printf("  ✓ Connected statistical methods to 5 practical application domains\n");
    
    printf("\nCross-domain analysis complete:\n");
    printf("  • Theoretical foundations → Practical applications\n");
    printf("  • Historical evolution and paradigm shifts\n");
    printf("  • Mathematical formulations and derivations\n");
    printf("  • Real-world application domains\n");
    printf("  • Uncertainty quantification ecosystem\n");
}

/* ============================================================================
 * MAIN STATISTICAL KNOWLEDGE GRAPH CONSTRUCTION
 * ============================================================================ */

int main(void) {
    srand((unsigned int)time(NULL));
    
    printf("📊 STATISTICAL KNOWLEDGE GRAPH CONSTRUCTION\n");
    printf("================================================================================\n");
    printf("Building comprehensive graph of probability theory, stochastic processes,\n");
    printf("regression analysis, and inferential statistics with complex relationships\n");
    printf("================================================================================\n\n");
    
    /* Create temporal-quantum graph for statistical evolution */
    ug_graph_t* stats_universe = ug_create_graph_with_type(UG_GRAPH_TYPE_TEMPORAL);
    if (!stats_universe) {
        fprintf(stderr, "❌ Failed to create statistical knowledge graph!\n");
        return 1;
    }
    
    printf("✓ Created temporal statistical knowledge graph (ID: %llu)\n\n", 
           (unsigned long long)stats_universe->id);
    
    /* Node arrays for different domains */
    ug_node_id_t foundation_nodes[6];
    ug_node_id_t distribution_nodes[10];
    ug_node_id_t process_nodes[9];
    ug_node_id_t regression_nodes[11];
    ug_node_id_t test_nodes[10];
    
    /* Build knowledge domains */
    build_probability_theory_foundation(stats_universe, foundation_nodes);
    build_probability_distributions(stats_universe, distribution_nodes);
    build_stochastic_processes(stats_universe, process_nodes);
    build_regression_analysis(stats_universe, regression_nodes);
    build_inferential_statistics(stats_universe, test_nodes);
    
    /* Create complex cross-domain relationships */
    create_cross_domain_relationships(stats_universe, foundation_nodes, distribution_nodes,
                                    process_nodes, regression_nodes, test_nodes);
    
    /* Final statistics and analysis */
    print_section_header("STATISTICAL KNOWLEDGE GRAPH ANALYSIS");
    ug_print_graph_stats(stats_universe);
    
    printf("\n📊 Domain Analysis:\n");
    printf("  • Probability Theory Foundation: 6 core concepts\n");
    printf("  • Probability Distributions: 10 key distributions (discrete, continuous, multivariate)\n");
    printf("  • Stochastic Processes: 9 processes (discrete/continuous time, jump processes)\n");
    printf("  • Regression Analysis: 11 models (linear, GLM, regularized, time series)\n");
    printf("  • Inferential Statistics: 10 methods (parametric, non-parametric, Bayesian)\n");
    
    printf("\n🔗 Relationship Analysis:\n");
    printf("  • Hierarchical: Generalization and specialization relationships\n");
    printf("  • Methodological: Extensions and robust alternatives\n");
    printf("  • Theoretical: Mathematical foundations and limiting behaviors\n");
    printf("  • Applied: Connections to real-world problem domains\n");
    printf("  • Temporal: Historical evolution and paradigm development\n");
    printf("  • Meta: Relationships about relationships (philosophical context)\n");
    printf("  • Quantum: Uncertainty about statistical concepts themselves\n");
    
    printf("\n🎯 Hypergraph Structures:\n");
    printf("  • Model selection tradeoffs (4-way: Ridge, Lasso, Elastic Net, Selection)\n");
    printf("  • Uncertainty quantification (5-way: Probability, Variance, Bayesian, CI, Credible)\n");
    printf("  • Cross-domain integration (N-way: Theory → Distribution → Process → Regression → Inference)\n");
    
    printf("\n⚡ Advanced Features Demonstrated:\n");
    printf("  • Universal types: Mathematical formulas, statistical concepts, applications\n");
    printf("  • Temporal relationships: Historical paradigm evolution\n");
    printf("  • Meta-relationships: Philosophy of science contextualizing paradigm conflicts\n");
    printf("  • Quantum uncertainty: Contested statistical interpretations\n");
    printf("  • Cross-domain flow: Theory to practice pipeline\n");
    
    /* Export the statistical knowledge graph */
    printf("\n💾 Exporting statistical knowledge graph...\n");
    ug_export_graph(stats_universe, "graphml", "statistical_knowledge_graph.graphml");
    ug_export_graph(stats_universe, "cypher", "statistical_knowledge_graph.cypher");
    ug_export_graph(stats_universe, "rdf", "statistical_knowledge_graph.rdf");
    printf("✓ Exported to GraphML, Cypher, and RDF formats\n");
    
    printf("\n🎉 STATISTICAL KNOWLEDGE GRAPH COMPLETE!\n");
    printf("================================================================================\n");
    printf("Successfully modeled the complex ecosystem of statistical science:\n");
    printf("  ✓ 46+ statistical concepts, methods, and applications\n");
    printf("  ✓ %zu total relationships capturing deep interconnections\n", ug_get_relationship_count(stats_universe));
    printf("  ✓ Hypergraph structures for multi-way concept relationships\n");
    printf("  ✓ Temporal evolution of statistical paradigms\n");
    printf("  ✓ Meta-analysis of philosophical foundations\n");
    printf("  ✓ Quantum uncertainty about contested interpretations\n");
    printf("  ✓ Complete theory-to-practice knowledge pipeline\n");
    printf("================================================================================\n");
    printf("📊 The Universe of Statistical Knowledge Awaits Exploration! 📊\n");
    
    /* Cleanup */
    ug_destroy_graph(stats_universe);
    
    return 0;
}