# DSPy Complete Cheat Sheet

## 📌 What is DSPy?

**DSPy** (Declarative Self-improving Python) is a framework from Stanford NLP for **programming—not prompting—language models**. It treats LLM interactions as modular, composable programs rather than brittle prompt strings.

### Key Philosophy
- **Program** the behavior instead of **prompting** for outputs
- Modular, composable, and optimizable AI systems
- Automatic prompt optimization and self-improvement
- Decouple logic from specific LLM implementation details

---

## 🚀 Quick Start

### Installation
```bash
pip install dspy-ai
```

### Basic Setup
```python
import dspy

# Configure Language Model
lm = dspy.LM('openai/gpt-4o-mini', api_key='your-api-key')
dspy.configure(lm=lm)

# Or for multiple providers
dspy.configure(lm=dspy.LM('anthropic/claude-sonnet-4'))
```

### Supported Providers (via LiteLLM)
- OpenAI: `openai/gpt-4o-mini`
- Anthropic: `anthropic/claude-sonnet-4`
- Google: `gemini/gemini-pro`
- Azure: `azure/<deployment_name>`
- Local: `openai/localhost:8000/v1` (for OpenAI-compatible endpoints)
- And 50+ more providers

---

## 📝 Core Concepts

### 1. Signatures
**Definition**: Declarative specifications of input/output behavior

#### Inline Signatures (Simple)
```python
# Basic: input -> output
"question -> answer"

# With types
"question: str -> answer: str"

# Multiple inputs/outputs
"context, question -> answer, confidence: float"

# With type constraints
"sentence -> sentiment: bool"
"question, choices: list[str] -> reasoning, selection: int"
```

#### Class-Based Signatures (Advanced)
```python
class EmotionClassifier(dspy.Signature):
    """Classify the emotion in a sentence."""
    
    sentence: str = dspy.InputField(desc="The input sentence to classify")
    emotion: Literal["happy", "sad", "angry", "neutral"] = dspy.OutputField(
        desc="The detected emotion"
    )
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")
```

#### Signature Manipulation
```python
# Add field
NewSig = MySig.append("extra_field", dspy.OutputField(desc="Extra info"))

# Remove field
NewSig = MySig.delete("temp_field")

# Insert at position
NewSig = MySig.insert(0, "context", dspy.InputField(desc="Context"))

# Prepend field
NewSig = MySig.prepend("prefix", dspy.InputField())

# Update instructions
NewSig = MySig.with_instructions("Translate to French.")
```

---

### 2. Modules
**Definition**: Building blocks that implement prompting techniques

#### Basic Module Pattern
```python
# 1. Declare module with signature
module = dspy.ModuleName('input -> output')

# 2. Call with input arguments
response = module(input="value")

# 3. Access outputs
print(response.output)
```

#### Core Modules

##### `dspy.Predict` - Basic Prediction
```python
# Simple prediction
classify = dspy.Predict('sentence -> sentiment: bool')
response = classify(sentence="Great product!")
print(response.sentiment)  # True

# Multiple completions
classify = dspy.Predict('sentence -> sentiment', n=5)
response = classify(sentence="Great!")
for completion in response.completions:
    print(completion.sentiment)
```

##### `dspy.ChainOfThought` - Reasoning Before Output
```python
qa = dspy.ChainOfThought('question -> answer')
response = qa(question="What is 2+2?")
print(response.reasoning)  # Step-by-step reasoning
print(response.answer)     # Final answer
```

##### `dspy.ChainOfThoughtWithHint` - Guided Reasoning
```python
qa = dspy.ChainOfThoughtWithHint('context, question -> answer')
response = qa(
    context="Paris is the capital of France",
    question="What is the capital of France?"
)
```

##### `dspy.ProgramOfThought` - Code Generation
```python
solver = dspy.ProgramOfThought('question -> answer: float')
response = solver(question="If I have 3 apples and buy 5 more, how many do I have?")
print(response.answer)  # 8.0
```

##### `dspy.ReAct` - Agent with Tools
```python
agent = dspy.ReAct('question -> answer', tools=[search_tool, calculator])
response = agent(question="What's the weather in Paris?")
```

##### `dspy.MultiChainComparison` - Compare Multiple Outputs
```python
comparer = dspy.MultiChainComparison('question -> answer', M=3)
response = comparer(question="What is AI?")
```

#### Custom Modules
```python
class RAGPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought('context, question -> answer')
    
    def forward(self, question):
        # Retrieve relevant docs
        context = self.retrieve(question).passages
        # Generate answer
        return self.generate(context=context, question=question)

# Usage
rag = RAGPipeline()
response = rag(question="What is DSPy?")
```

---

## 🎯 Optimizers (formerly Teleprompters)

Optimizers automatically improve your program by tuning prompts, selecting examples, or fine-tuning weights.

### Available Optimizers

#### 1. `BootstrapFewShot` - Few-Shot Learning
```python
from dspy.teleprompt import BootstrapFewShot

# Define metric
def validate_answer(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

# Compile
optimizer = BootstrapFewShot(metric=validate_answer, max_bootstrapped_demos=4)
optimized_program = optimizer.compile(
    student=program,
    trainset=trainset
)
```

#### 2. `BootstrapFewShotWithRandomSearch` - With Random Search
```python
optimizer = BootstrapFewShotWithRandomSearch(
    metric=validate_answer,
    max_bootstrapped_demos=4,
    num_candidate_programs=8
)
optimized = optimizer.compile(program, trainset=trainset)
```

#### 3. `MIPROv2` - Multi-Stage Instruction Optimization
```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    metric=validate_answer,
    num_candidates=10,
    init_temperature=1.0
)
optimized = optimizer.compile(
    program,
    trainset=trainset,
    valset=valset,  # Optional validation set
    num_trials=100
)
```

#### 4. `BootstrapFinetune` - Fine-Tune Model Weights
```python
from dspy.teleprompt import BootstrapFinetune

optimizer = BootstrapFinetune(metric=validate_answer)
optimized = optimizer.compile(
    student=program,
    trainset=trainset,
    target='gpt-3.5-turbo'
)
```

#### 5. `COPRO` - Coordinate Ascent Prompt Optimization
```python
from dspy.teleprompt import COPRO

optimizer = COPRO(
    metric=validate_answer,
    breadth=10,
    depth=3
)
optimized = optimizer.compile(program, trainset=trainset)
```

#### 6. `GEPA` - Grounded Evolutionary Prompt Adaptation
```python
from dspy.teleprompt import GEPA

optimizer = GEPA(metric=validate_answer, num_iterations=10)
optimized = optimizer.compile(program, trainset=trainset)
```

---

## 📊 Metrics

Define success criteria for optimization:

```python
# Binary metric
def exact_match(example, pred, trace=None):
    return example.answer == pred.answer

# Continuous metric
def f1_score(example, pred, trace=None):
    # Calculate F1 between example.answer and pred.answer
    # Return float between 0 and 1
    return calculate_f1(example.answer, pred.answer)

# Multiple criteria
def complex_metric(example, pred, trace=None):
    correct = example.answer.lower() in pred.answer.lower()
    length_ok = len(pred.answer.split()) <= 50
    return correct and length_ok
```

### Built-in Metrics
```python
from dspy.evaluate import SemanticF1

# Semantic similarity metric
metric = SemanticF1()
score = metric(example, prediction)
```

---

## 🔍 Retrieval Integration

### Built-in Retrievers

#### ColBERTv2
```python
colbert = dspy.ColBERTv2(url='http://localhost:8893/api/search')
dspy.configure(rm=colbert)

# In module
retriever = dspy.Retrieve(k=5)
results = retriever(query="What is machine learning?")
print(results.passages)
```

#### Custom Retriever
```python
class CustomRetriever(dspy.Retrieve):
    def __init__(self, k=3):
        super().__init__(k=k)
    
    def forward(self, query: str) -> dspy.Prediction:
        # Your retrieval logic
        passages = my_search_function(query, k=self.k)
        return dspy.Prediction(passages=passages)
```

---

## 🛠️ Advanced Features

### 1. Assertions and Constraints
```python
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

class AnswerWithConstraints(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought('question -> answer')
    
    def forward(self, question):
        pred = self.generate(question=question)
        
        # Add assertion
        dspy.Suggest(
            len(pred.answer.split()) <= 20,
            "Answer should be less than 20 words"
        )
        
        return pred

# Wrap with backtracking
constrained_module = assert_transform_module(
    AnswerWithConstraints(),
    backtrack_handler
)
```

### 2. LM Usage Tracking
```python
# Reset tracking
dspy.reset_tracking()

# Run program
response = program(input="test")

# Get usage stats
usage = dspy.get_usage()
print(usage)
# Output: {'gpt-4o-mini': {'prompt_tokens': 150, 'completion_tokens': 50, 'cost': 0.001}}
```

### 3. Caching
```python
# DSPy automatically caches LM calls
# Clear cache if needed
lm.clear_cache()
```

### 4. Multi-Model Programs
```python
# Different LMs for different modules
fast_lm = dspy.LM('openai/gpt-3.5-turbo')
smart_lm = dspy.LM('openai/gpt-4')

class HybridProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict('text -> category')
        self.analyze = dspy.ChainOfThought('text, category -> analysis')
    
    def forward(self, text):
        # Use fast model for classification
        with dspy.context(lm=fast_lm):
            category = self.classify(text=text).category
        
        # Use smart model for analysis
        with dspy.context(lm=smart_lm):
            analysis = self.analyze(text=text, category=category).analysis
        
        return dspy.Prediction(category=category, analysis=analysis)
```

---

## 💡 Common Patterns

### 1. Simple Q&A
```python
qa = dspy.ChainOfThought('question -> answer')
response = qa(question="What is Python?")
print(response.answer)
```

### 2. Classification
```python
class Classifier(dspy.Signature):
    """Classify sentiment of text."""
    text: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()

classifier = dspy.Predict(Classifier)
result = classifier(text="I love this product!")
```

### 3. RAG (Retrieval-Augmented Generation)
```python
class SimpleRAG(dspy.Module):
    def __init__(self, k=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=k)
        self.generate = dspy.ChainOfThought('context, question -> answer')
    
    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)
```

### 4. Multi-Hop Reasoning
```python
class MultiHopQA(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = dspy.ChainOfThought('context, question -> search_query')
        self.generate_answer = dspy.ChainOfThought('context, question -> answer')
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_query(
                context=context,
                question=question
            ).search_query
            
            passages = self.retrieve(query).passages
            context.extend(passages)
        
        return self.generate_answer(context=context, question=question)
```

### 5. Self-Consistency
```python
class SelfConsistentQA(dspy.Module):
    def __init__(self, num_samples=5):
        super().__init__()
        self.qa = dspy.ChainOfThought('question -> answer', n=num_samples)
    
    def forward(self, question):
        responses = self.qa(question=question).completions
        # Find most common answer
        answers = [r.answer for r in responses]
        most_common = max(set(answers), key=answers.count)
        return dspy.Prediction(answer=most_common)
```

### 6. Agent with Tools
```python
def calculator(expression: str) -> float:
    """Evaluate mathematical expression."""
    return eval(expression)

def search(query: str) -> str:
    """Search the web."""
    # Your search implementation
    return search_results

agent = dspy.ReAct(
    'question -> answer',
    tools=[calculator, search]
)

response = agent(question="What is the GDP of France times 2?")
```

---

## 📈 Evaluation

### Evaluate Program Performance
```python
from dspy.evaluate import Evaluate

# Define evaluator
evaluator = Evaluate(
    devset=test_set,
    metric=exact_match,
    num_threads=4,
    display_progress=True
)

# Evaluate
score = evaluator(program)
print(f"Accuracy: {score}")
```

### Compare Programs
```python
baseline_score = evaluator(baseline_program)
optimized_score = evaluator(optimized_program)

print(f"Improvement: {(optimized_score - baseline_score) / baseline_score * 100:.1f}%")
```

---

## 🔧 Configuration Options

### Global Configuration
```python
dspy.configure(
    lm=dspy.LM('openai/gpt-4o-mini'),
    rm=colbert_retriever,  # Optional retriever
    temperature=0.7,
    max_tokens=150
)
```

### Per-Module Configuration
```python
# Configure specific module
predict = dspy.Predict('question -> answer', temperature=0.9, max_tokens=100)

# Or in class
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(
            'question -> answer',
            temperature=0.5
        )
```

### Context Manager for Temporary Config
```python
with dspy.context(lm=different_lm, temperature=1.0):
    response = module(input="test")
```

---

## 🎓 Best Practices

### 1. Start Simple
```python
# Begin with basic modules
qa = dspy.Predict('question -> answer')

# Add complexity as needed
qa = dspy.ChainOfThought('question -> answer')
```

### 2. Use Meaningful Field Names
```python
# Bad
"a -> b"

# Good
"customer_review -> sentiment_score: float"
```

### 3. Provide Descriptions for Complex Tasks
```python
class ComplexTask(dspy.Signature):
    """Analyze customer feedback and extract actionable insights."""
    
    feedback: str = dspy.InputField(desc="Raw customer feedback text")
    sentiment: str = dspy.OutputField(desc="Positive, Negative, or Neutral")
    key_issues: list[str] = dspy.OutputField(desc="List of main concerns")
    priority: int = dspy.OutputField(desc="Priority level 1-5")
```

### 4. Use Optimizers for Production
```python
# Don't rely on default prompts
# Always optimize for your specific task and data
optimizer = MIPROv2(metric=your_metric)
optimized_program = optimizer.compile(program, trainset=trainset)
```

### 5. Monitor Performance
```python
# Track LM usage
dspy.reset_tracking()
result = program(input)
usage = dspy.get_usage()
print(f"Cost: ${usage['gpt-4o-mini']['cost']}")
```

---

## 🐛 Debugging

### Inspect Prompts
```python
# See what prompt DSPy generates
lm.inspect_history(n=1)  # Last call
```

### Verbose Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Module State
```python
# Inspect module parameters
print(module.dump_state())

# Load saved state
module.load_state(saved_state)
```

---

## 📦 Save & Load

### Save Compiled Program
```python
# Save
optimized_program.save('path/to/model')

# Load
loaded_program = MyProgramClass()
loaded_program.load('path/to/model')
```

### Export Prompts
```python
# Extract optimized prompts
prompts = program.get_prompts()
for module_name, prompt in prompts.items():
    print(f"{module_name}: {prompt}")
```

---

## 🚨 Common Pitfalls

### 1. Not Using Optimizers
```python
# ❌ Don't use unoptimized programs in production
program = BasicQA()

# ✅ Always optimize
optimizer = BootstrapFewShot(metric=metric)
program = optimizer.compile(BasicQA(), trainset=trainset)
```

### 2. Ignoring Signatures
```python
# ❌ Vague signatures
"input -> output"

# ✅ Descriptive signatures
"customer_complaint -> root_cause, suggested_action"
```

### 3. Not Testing with Real Data
```python
# ✅ Always evaluate on held-out test set
test_score = evaluator(program, devset=test_set)
```

### 4. Over-constraining Output
```python
# ❌ Too restrictive
answer: Literal["A", "B", "C", "D"]  # What if answer is "E"?

# ✅ Appropriate constraints
answer: str = dspy.OutputField(desc="Choose from A, B, C, or D")
```

---

## 📚 Quick Reference

### Module Hierarchy
```
dspy.Module (Base)
├── dspy.Predict (Basic)
├── dspy.ChainOfThought (Reasoning)
├── dspy.ChainOfThoughtWithHint (Guided)
├── dspy.ProgramOfThought (Code)
├── dspy.ReAct (Agent)
├── dspy.MultiChainComparison (Compare)
└── Custom Modules
```

### Optimizer Comparison

| Optimizer | Use Case | Speed | Quality |
|-----------|----------|-------|---------|
| BootstrapFewShot | Quick few-shot learning | Fast | Good |
| BootstrapFewShotWithRandomSearch | Better few-shot | Medium | Better |
| MIPROv2 | Best prompt tuning | Slow | Best |
| BootstrapFinetune | Model fine-tuning | Slowest | Best |
| COPRO | Coordinate optimization | Medium | Good |

### Common Signatures

```python
# Q&A
"question -> answer"

# Classification  
"text -> category, confidence: float"

# Summarization
"document -> summary"

# Translation
"english_text -> french_text"

# Sentiment
"review -> sentiment: bool, reasoning"

# Extraction
"text -> entities: list[str]"

# Multi-hop
"context, question -> search_query"
"contexts: list[str], question -> answer"
```

---

## 🌟 Real-World Example: Complete RAG System

```python
import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate

# 1. Setup
lm = dspy.LM('openai/gpt-4o-mini')
colbert = dspy.ColBERTv2(url='http://localhost:8893/api/search')
dspy.configure(lm=lm, rm=colbert)

# 2. Define RAG Program
class OptimizedRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=5)
        self.generate = dspy.ChainOfThought('context, question -> answer')
    
    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# 3. Create Training Data
trainset = [
    dspy.Example(
        question="What is machine learning?",
        answer="Machine learning is a subset of AI..."
    ).with_inputs('question'),
    # ... more examples
]

# 4. Define Metric
def answer_quality(example, pred, trace=None):
    # Check if answer is relevant and accurate
    return (
        example.answer.lower() in pred.answer.lower() or
        pred.answer.lower() in example.answer.lower()
    )

# 5. Optimize
optimizer = MIPROv2(metric=answer_quality, num_candidates=10)
optimized_rag = optimizer.compile(
    OptimizedRAG(),
    trainset=trainset,
    num_trials=50
)

# 6. Evaluate
evaluator = Evaluate(devset=test_set, metric=answer_quality)
score = evaluator(optimized_rag)
print(f"Test Score: {score}")

# 7. Use in Production
response = optimized_rag(question="How does backpropagation work?")
print(response.answer)

# 8. Save
optimized_rag.save('models/optimized_rag')
```

---

## 🔗 Resources

- **GitHub**: https://github.com/stanfordnlp/dspy
- **Documentation**: https://dspy.ai
- **Discord**: Join the DSPy community
- **Paper**: "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"

---

## 📝 Summary

DSPy transforms LLM application development by:
1. **Replacing prompts with signatures** - Declare what you want, not how to ask
2. **Using modules** - Compose reusable building blocks
3. **Optimizing automatically** - Let algorithms find the best prompts
4. **Maintaining modularity** - Build complex systems from simple components

**Remember**: DSPy is about *programming* behavior, not *prompting* for outputs!
