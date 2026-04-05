# DSPy-Slim Formal Specification

## 1. Type Universe

### 1.1. Primitive Scalars

```typescript
type FieldKind = "input" | "output";
type TypeTag = "str" | "int" | "float" | "bool" | "list" | "dict" | "literal" | "enum" | "optional" | "union" | "custom";
type Role = "system" | "user" | "assistant" | "developer";
type ModelType = "chat" | "responses";
type AdapterKind = "chat" | "json";
```

### 1.2. Field

```typescript
interface Field {
  kind: FieldKind;
  name: string;
  type_tag: TypeTag;
  type_args: TypeTag[];          // e.g. list<str> → type_tag="list", type_args=["str"]
  description: string;
  prefix: string;                // inferred from name via infer_prefix
  constraints: string[];         // human-readable pydantic constraint strings
  default: unknown | undefined;  // undefined ≡ required
  is_type_undefined: boolean;    // true when user omitted explicit annotation
}
```

**Invariants:**

- ∀ f ∈ Signature.fields: f.kind ∈ {"input", "output"}
- ∀ f ∈ Signature.fields: f.prefix ≠ ""
- ∀ f ∈ Signature.fields: f.name matches `[a-zA-Z_][a-zA-Z0-9_]*`

### 1.3. Signature

```typescript
interface Signature {
  name: string;
  instructions: string;
  input_fields: OrderedMap<string, Field>;   // insertion-ordered
  output_fields: OrderedMap<string, Field>;  // insertion-ordered
}
```

**Derived:**

```typescript
// fields = input_fields ∪ output_fields (inputs first)
// signature_string = join(input_fields.keys, ", ") + " -> " + join(output_fields.keys, ", ")
```

**Operations:**

```typescript
interface SignatureOps {
  append(sig: Signature, name: string, field: Field, type_tag: TypeTag): Signature;
  prepend(sig: Signature, name: string, field: Field, type_tag: TypeTag): Signature;
  delete(sig: Signature, field_name: string): Signature;
  with_instructions(sig: Signature, instructions: string): Signature;
  with_updated_fields(sig: Signature, field_name: string, field: Field): Signature;
  equals(a: Signature, b: Signature): boolean;
}
```

**Algebraic Laws:**

```
append(sig, n, f, t).output_fields.last_key == n
prepend(sig, n, f, t).output_fields.first_key == n
delete(append(sig, n, f, t), n) ≡ sig
|sig.input_fields ∩ sig.output_fields| == 0          // disjoint key sets
```

### 1.4. Signature Parsing

Input: string of form `"field1, field2: type -> field3, field4: type"`

```typescript
interface ParsedField {
  name: string;
  type_tag: TypeTag;
  is_type_undefined: boolean;
}

interface ParseResult {
  inputs: ParsedField[];
  outputs: ParsedField[];
}
```

**Constraints:**

- Input string contains exactly one `"->"` separator
- `inputs ∩ outputs == ∅` (field names disjoint)
- Untyped fields default to `type_tag: "str"`, `is_type_undefined: true`

### 1.5. `infer_prefix`

Pure function: `string → string`

```
infer_prefix("camelCaseText")    == "Camel Case Text"
infer_prefix("snake_case_text")  == "Snake Case Text"
infer_prefix("text2number")      == "Text 2 Number"
infer_prefix("HTMLParser")       == "HTML Parser"
```

Algorithm:

1. Insert `_` before `[A-Z][a-z]+` preceded by `.`
2. Insert `_` between `[a-z0-9]` and `[A-Z]`
3. Insert `_` between `[a-zA-Z]` and `\d`, and between `\d` and `[a-zA-Z]`
4. Split on `_`
5. Map: all-uppercase words preserved; others capitalized
6. Join with `" "`

---

## 2. Data Containers

### 2.1. Example

```typescript
interface Example {
  _store: Map<string, unknown>;
  _input_keys: Set<string> | null;
}
```

**Operations:**

```typescript
interface ExampleOps {
  // Construction
  create(base?: Example | Record<string, unknown>, overrides?: Record<string, unknown>): Example;

  // Field access
  get(e: Example, key: string): unknown;           // throws on missing
  get_or(e: Example, key: string, def: unknown): unknown;
  set(e: Example, key: string, value: unknown): void;
  has(e: Example, key: string): boolean;
  keys(e: Example, include_dspy?: boolean): string[];
  len(e: Example): number;                         // excludes dspy_ prefixed keys

  // Splitting
  with_inputs(e: Example, ...keys: string[]): Example;  // returns copy
  inputs(e: Example): Example;    // requires _input_keys ≠ null
  labels(e: Example): Example;    // complement of inputs

  // Serialization
  to_dict(e: Example): Record<string, unknown>;

  // Copy
  copy(e: Example, overrides?: Record<string, unknown>): Example;
  without(e: Example, ...keys: string[]): Example;
}
```

**Invariants:**

- `inputs(e) ∪ labels(e) ⊇ {k ∈ keys(e) | k ∈ _input_keys}`
- `inputs(e).keys ∩ labels(e).keys == ∅`
- `len(e)` excludes keys matching `^dspy_`
- `with_inputs` returns a **copy**; original is unmodified

### 2.2. Prediction

```typescript
interface Prediction extends Example {
  _completions: Completions | null;
  // _input_keys: deleted (attribute absent, not null)
  // _demos: deleted (attribute absent, not null)
}

interface Completions {
  _completions: Map<string, unknown[]>;   // all arrays same length
  signature: Signature | null;
}
```

**Construction:**

```typescript
// from_completions(list_or_dict, signature?) → Prediction
// Prediction._store = { k: v[0] for (k, v) in completions.items }
```

**Numeric protocol** (requires `score` field):

```
float(p)       → p._store["score"] as float

p + x          → float(p) + float(x)     where x ∈ {Prediction, int, float}
x + p          → float(x) + float(p)     where x ∈ {Prediction, int, float}

p / x          → float(p) / float(x)     where x ∈ {Prediction, int, float}
x / p          → float(x) / float(p)     where x ∈ {Prediction, int, float}

p < x          → float(p) < float(x)     where x ∈ {Prediction, int, float}
p <= x, p > x, p >= x                    (analogous)
```

---

## 3. Module System

### 3.1. Parameter

```typescript
// Marker interface. No fields, no methods.
interface Parameter {}
```

### 3.2. BaseModule

```typescript
interface BaseModule {
  named_parameters(): Array<[string, Parameter]>;
  parameters(): Parameter[];
  deepcopy(): BaseModule;
  reset_copy(): BaseModule;
}
```

**`named_parameters` traversal algorithm:**

```
visited: Set<ObjectId> = ∅
named_parameters: Array<[string, Parameter]> = []

add_parameter(name, value):
  if value is Parameter:                             // ⚠ if/elif: Parameter takes precedence over Module
    if id(value) ∉ visited:
      visited.add(id(value))
      named_parameters.push((name, value))
  elif value is Module:                              // only reached if NOT Parameter
    if not getattr(value, "_compiled", false):
      for (sub_name, param) in value.named_parameters():
        add_parameter(name + "." + sub_name, param)

if self is Parameter:
  add_parameter("self", self)

for (name, value) in self.__dict__:
  if value is Parameter:
    add_parameter(name, value)
  elif value is Module:                              // only reached if NOT Parameter
    if not getattr(value, "_compiled", false):
      for (sub_name, param) in value.named_parameters():
        add_parameter(name + "." + sub_name, param)
  elif value is list|tuple:
    for each (idx, item): add_parameter(f"{name}[{idx}]", item)
  elif value is dict:
    for each (key, item): add_parameter(f"{name}['{key}']", item)
```

**⚠ Port note:** `Predict` is both `Parameter` and `Module`. The `if/elif` precedence
ensures it is emitted as a Parameter and **not** recursively traversed as a Module.

### 3.3. Module

```typescript
interface Module extends BaseModule {
  _compiled: boolean;
  callbacks: Callback[];
  history: HistoryEntry[];

  // Abstract
  forward(...kwargs: Record<string, unknown>): Prediction;
  aforward(...kwargs: Record<string, unknown>): Promise<Prediction>;

  // Derived
  named_predictors(): Array<[string, Predict]>;
  predictors(): Predict[];
  set_lm(lm: BaseLM): void;
  get_lm(): BaseLM;
}
```

**Call protocol:**

```
Module.__call__(kwargs):
  caller_modules = settings.caller_modules ∪ {self}
  with settings.context(caller_modules):
    return self.forward(kwargs)
```

**Invariant:** `named_predictors() ⊆ named_parameters()` — filters to instances of `Predict`.

---

## 4. Predict Pipeline

### 4.1. Predict

```typescript
interface Predict extends Module, Parameter {
  signature: Signature;
  config: Record<string, unknown>;
  lm: BaseLM | null;
  demos: Demo[];
  traces: Trace[];

  // Inherited from Parameter
  reset(): void;  // lm=null, traces=[], train=[], demos=[]
}
```

### 4.2. Forward Pipeline

```typescript
// Predict.forward(kwargs) executes this pipeline:

interface PredictPipeline {
  // Phase 1: Preprocess
  resolve_signature(kwargs: Record<string, unknown>): Signature;
  resolve_demos(kwargs: Record<string, unknown>): Demo[];
  resolve_config(kwargs: Record<string, unknown>): Record<string, unknown>;
  resolve_lm(kwargs: Record<string, unknown>): BaseLM;      // lm ∈ {kwargs.lm, self.lm, settings.lm}
  populate_defaults(sig: Signature, kwargs: Record<string, unknown>): Record<string, unknown>;
  validate_inputs(sig: Signature, kwargs: Record<string, unknown>): void;  // warn on missing/extra

  // Phase 2: Adapt + Call
  format(adapter: Adapter, sig: Signature, demos: Demo[], inputs: Record<string, unknown>): Message[];
  call_lm(lm: BaseLM, messages: Message[], config: Record<string, unknown>): LMOutput[];
  parse(adapter: Adapter, sig: Signature, completion: string): Record<string, unknown>;

  // Phase 3: Postprocess
  build_prediction(completions: Record<string, unknown>[], sig: Signature): Prediction;
  append_trace(predict: Predict, inputs: Record<string, unknown>, pred: Prediction): void;
}
```

**LM resolution order:** `kwargs["lm"] > self.lm > settings.lm`

**Temperature auto-adjust:**

```
effective_temperature :=
  config.get("temperature")          // falsy → fallback
  OR lm.kwargs.get("temperature")    // Python `or` semantics: 0 is falsy

effective_n :=
  config.get("n")
  OR lm.kwargs.get("n")
  OR lm.kwargs.get("num_generations")
  OR 1                                // default

if (effective_temperature is null OR effective_temperature <= 0.15) AND effective_n > 1:
  config["temperature"] = 0.7
```

**⚠ Port note:** Python `or` uses truthiness (0 is falsy), not nullish coalescing.

### 4.3. ChainOfThought

```typescript
interface ChainOfThought extends Module {
  predict: Predict;    // signature = original_signature.prepend("reasoning", OutputField, str)
}
// forward(kwargs) = self.predict(kwargs)
```

---

## 5. Adapter Contract

### 5.1. Message Wire Format

```typescript
interface Message {
  role: Role;
  content: string | ContentPart[];
}

interface ContentPart {
  type: "text" | "image_url" | "file";
  text?: string;
  image_url?: { url: string };
  file?: { file_data?: string; filename?: string; file_id?: string };
}
```

### 5.2. Adapter Interface

```typescript
interface Adapter {
  use_native_function_calling: boolean;

  __call__(lm: BaseLM, lm_kwargs: Record<string, unknown>, sig: Signature, demos: Demo[], inputs: Record<string, unknown>): Record<string, unknown>[];
  acall(lm: BaseLM, lm_kwargs: Record<string, unknown>, sig: Signature, demos: Demo[], inputs: Record<string, unknown>): Promise<Record<string, unknown>[]>;

  format(sig: Signature, demos: Demo[], inputs: Record<string, unknown>): Message[];
  parse(sig: Signature, completion: string): Record<string, unknown>;

  format_system_message(sig: Signature): string;
  format_field_description(sig: Signature): string;
  format_field_structure(sig: Signature): string;
  format_task_description(sig: Signature): string;
  format_user_message_content(sig: Signature, inputs: Record<string, unknown>, prefix?: string, suffix?: string, main_request?: boolean): string;
  format_assistant_message_content(sig: Signature, outputs: Record<string, unknown>, missing_field_message?: string): string;
  format_demos(sig: Signature, demos: Demo[]): Message[];
}
```

### 5.3. Adapter Pipeline

```
__call__(lm, lm_kwargs, sig, demos, inputs):
  processed_sig = _call_preprocess(lm, lm_kwargs, sig, inputs)    // tool extraction, native types
  messages = format(processed_sig, demos, inputs)
  outputs = lm(messages, **lm_kwargs)
  return _call_postprocess(processed_sig, sig, outputs, lm, lm_kwargs)
```

### 5.4. ChatAdapter Parse Protocol

Delimiter pattern: `[[ ## field_name ## ]]`

```
parse(sig, completion):
  current_header = null
  current_lines = []
  sections: OrderedMap<string, string> = {}

  for each line in completion.splitlines():
    match = /^\[\[ ## (\w+) ## \]\]/.match(line.strip())
    if match:
      if current_header ≠ null ∧ current_header ∈ sig.output_fields ∧ current_header ∉ sections:
        sections[current_header] = join(current_lines, "\n").strip()
      current_header = match[1]
      current_lines = []
      remaining = line[match.end:].strip()
      if remaining ≠ "": current_lines.push(remaining)
    else:
      current_lines.push(line)

  // flush last section
  if current_header ≠ null ∧ current_header ∈ sig.output_fields ∧ current_header ∉ sections:
    sections[current_header] = join(current_lines, "\n").strip()

  assert sections.keys == sig.output_fields.keys    // raises AdapterParseError
  for each (k, v) in sections:
    fields[k] = parse_value(v, sig.output_fields[k].annotation)
  return fields
```

**Constraints:**
- Header must start the stripped line (not mid-line)
- First occurrence of a field wins; duplicates are ignored
- Non-output-field headers (e.g. `completed`) are discarded

### 5.5. JSONAdapter Parse Protocol

```
parse(sig, completion):
  fields = json_repair.loads(completion)
  if not dict: extract first recursive JSON object via regex
  filter to keys ∈ sig.output_fields
  cast each value via parse_value(v, annotation)
  assert fields.keys == sig.output_fields.keys
  return fields
```

### 5.6. Message Assembly Order

```
messages = [
  { role: "system", content: format_system_message(sig) },
  ...format_demos(sig, demos),           // user/assistant pairs
  ...format_conversation_history(...),    // if History field present
  { role: "user",  content: format_user_message_content(sig, inputs, main_request=true) }
]
```

---

## 6. Language Model Contract

### 6.1. BaseLM

```typescript
interface BaseLM {
  model: string;
  model_type: ModelType;
  cache: boolean;
  kwargs: Record<string, unknown>;    // default params: temperature, max_tokens, ...
  history: HistoryEntry[];

  // Capability predicates
  readonly supports_function_calling: boolean;
  readonly supports_reasoning: boolean;
  readonly supports_response_schema: boolean;
  readonly supported_params: Set<string>;

  // Core
  forward(prompt?: string, messages?: Message[]): LMResponse;
  aforward(prompt?: string, messages?: Message[]): Promise<LMResponse>;

  // Response processing
  __call__(prompt?: string, messages?: Message[]): LMOutput[];
  acall(prompt?: string, messages?: Message[]): Promise<LMOutput[]>;
  copy(overrides?: Record<string, unknown>): BaseLM;
}
```

### 6.2. LMResponse (OpenAI Chat Completion shape)

```typescript
interface LMResponse {
  choices: Array<{
    message: {
      content: string | null;
      tool_calls?: ToolCallWire[];
    };
    finish_reason: string;
    logprobs?: unknown;
  }>;
  usage: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
  model: string;
}
```

### 6.3. LMResponse (OpenAI Responses shape)

```typescript
interface LMResponseResponses {
  output: Array<{
    type: "message" | "function_call";
    content?: Array<{ text: string }>;
    name?: string;
    arguments?: string;
  }>;
  usage: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
  model: string;
}
```

### 6.4. LMOutput (processed)

```typescript
type LMOutput = string | {
  text: string;
  tool_calls?: ToolCallWire[];
  logprobs?: unknown;
  citations?: unknown[];
};
```

### 6.5. HistoryEntry

```typescript
interface HistoryEntry {
  prompt: string | null;
  messages: Message[] | null;
  kwargs: Record<string, unknown>;
  response: LMResponse;
  outputs: LMOutput[];
  usage: Record<string, number>;
  cost: number | null;
  timestamp: string;          // ISO 8601
  uuid: string;
  model: string;
  response_model: string;
  model_type: ModelType;
}
```

### 6.6. History Update Protocol

```
update_history(entry):
  if settings.disable_history: return           // disables ALL history

  // Global history — always updated (hardcoded cap, not affected by max_history_size)
  if GLOBAL_HISTORY.len >= MAX_HISTORY_SIZE:     // MAX_HISTORY_SIZE = 10000 (constant)
    GLOBAL_HISTORY.shift()                       // drop oldest
  GLOBAL_HISTORY.push(entry)

  if settings.max_history_size == 0: return      // disables per-LM and per-module only

  // Per-LM history
  if self.history.len >= settings.max_history_size:
    self.history.shift()
  self.history.push(entry)

  // Per-caller-module history
  for module in settings.caller_modules:
    if module.history.len >= settings.max_history_size:
      module.history.shift()
    module.history.push(entry)
```

**⚠ Port note:** `max_history_size == 0` does NOT disable global history; only `disable_history` does.

---

## 7. Tool System

### 7.1. Tool

```typescript
interface Tool {
  func: Callable;
  name: string;
  desc: string;
  args: Record<string, JSONSchema>;
  arg_types: Record<string, TypeTag>;
  has_kwargs: boolean;

  __call__(kwargs: Record<string, unknown>): unknown;
  acall(kwargs: Record<string, unknown>): Promise<unknown>;
  format_as_openai_function_call(): OpenAIFunctionCallSchema;
}
```

### 7.2. ToolCalls

```typescript
interface ToolCall {
  name: string;
  args: Record<string, unknown>;
}

interface ToolCalls {
  tool_calls: ToolCall[];
}
```

### 7.3. History (Conversation)

```typescript
interface History {
  messages: Array<Record<string, unknown>>;   // frozen, validated
}
```

---

## 8. ReAct Agent

### 8.1. Structure

```typescript
interface ReAct extends Module {
  signature: Signature;
  max_iters: number;
  tools: Map<string, Tool>;         // includes synthetic "finish" tool
  react: Predict;                   // react_signature
  extract: ChainOfThought;          // fallback_signature
}
```

### 8.2. react_signature

```
inputs: [...original_sig.input_fields, trajectory: str]
outputs: [next_thought: str, next_tool_name: Literal[...tool_names], next_tool_args: dict[str, Any]]
instructions: agent-loop preamble + tool descriptions
```

### 8.3. fallback_signature

```
inputs: [...original_sig.input_fields, ...original_sig.output_fields, trajectory: str]
instructions: original_sig.instructions
```

### 8.4. ReAct State Machine

See §10.2.

---

## 9. Callback System

### 9.1. Callback Interface

```typescript
interface Callback {
  on_module_start(call_id: string, instance: Module, inputs: Record<string, unknown>): void;
  on_module_end(call_id: string, outputs: unknown | null, exception: Error | null): void;
  on_lm_start(call_id: string, instance: BaseLM, inputs: Record<string, unknown>): void;
  on_lm_end(call_id: string, outputs: unknown | null, exception: Error | null): void;
  on_adapter_format_start(call_id: string, instance: Adapter, inputs: Record<string, unknown>): void;
  on_adapter_format_end(call_id: string, outputs: unknown | null, exception: Error | null): void;
  on_adapter_parse_start(call_id: string, instance: Adapter, inputs: Record<string, unknown>): void;
  on_adapter_parse_end(call_id: string, outputs: unknown | null, exception: Error | null): void;
  on_tool_start(call_id: string, instance: Tool, inputs: Record<string, unknown>): void;
  on_tool_end(call_id: string, outputs: unknown | null, exception: Error | null): void;
}
```

### 9.2. Callback Dispatch

```
dispatch_type(instance, fn_name):
  if instance is BaseLM       → lm
  if instance is Evaluate      → evaluate
  if instance is Adapter:
    if fn_name == "format"     → adapter_format
    if fn_name == "parse"      → adapter_parse
  if instance is Tool          → tool
  else                         → module
```

### 9.3. `with_callbacks` Protocol

```
with_callbacks(fn)(instance, *args, **kwargs):
  callbacks = settings.callbacks ++ instance.callbacks
  if callbacks == []: return fn(instance, *args, **kwargs)
  call_id = new_uuid()
  for cb in callbacks: cb.on_{type}_start(call_id, instance, inputs)
  parent_call_id = ACTIVE_CALL_ID.get()
  ACTIVE_CALL_ID.set(call_id)
  try:
    result = fn(instance, *args, **kwargs)
    return result
  except e:
    exception = e; raise
  finally:
    ACTIVE_CALL_ID.set(parent_call_id)
    for cb in callbacks: cb.on_{type}_end(call_id, result, exception)
```

---

## 10. State Machines

### 10.1. Predict Forward State Machine

```
States: {Init, Preprocess, Format, LMCall, Parse, Postprocess, Done, Error}

Transitions:
  Init        → Preprocess     [always]
  Preprocess  → Error          [lm == null ∨ lm is string ∨ lm not BaseLM]
  Preprocess  → Format         [lm valid]
  Format      → LMCall         [messages produced]
  Format      → Error          [adapter format failure]
  LMCall      → Parse          [lm returns outputs]
  LMCall      → Error          [ContextWindowExceeded ∨ BadRequest]
  Parse       → Postprocess    [fields.keys == sig.output_fields.keys]
  Parse       → Error          [AdapterParseError]
  Postprocess → Done           [prediction built, trace appended]
```

### 10.2. ReAct Forward State Machine

```
States: {Init, Call_React, Process_Tool, Call_Extract, Done, Error}
Variables: iteration: 0..MAX_ITERS, trajectory: Map<string, string>

Transitions:
  Init          → Call_React      [iteration=0, trajectory=∅]
  Call_React    → Process_Tool    [pred.next_tool_name ≠ "finish"]
  Call_React    → Call_Extract    [pred.next_tool_name == "finish"]
  Call_React    → Call_Extract    [ValueError from invalid tool]
  Process_Tool  → Call_React      [iteration < max_iters, observation appended]
  Process_Tool  → Call_Extract    [iteration ≥ max_iters]
  Call_Extract  → Done            [extraction successful]
  Call_Extract  → Error           [extraction fails after truncation retries]

Context Window Truncation Sub-Machine (applies to Call_React and Call_Extract):
  Up to 3 LM invocations are attempted (call_index: 0..2).
  After each ContextWindowExceeded, truncate oldest 4 trajectory keys and retry.
  If all 3 invocations fail with ContextWindowExceeded, raise error.

  CallLM        → Success         [no ContextWindowExceeded]
  CallLM        → Truncate        [ContextWindowExceeded ∧ call_index < 2]
  Truncate      → CallLM          [drop oldest 4 trajectory keys]
  CallLM        → Error           [ContextWindowExceeded ∧ call_index == 2]
  Truncate      → Error           [trajectory.len < 4]
```

### 10.3. Parallel Execution State Machine

```
States: {Idle, Running, Cancelled, Done, Error}
Per-task states: {Pending, Executing, Completed, Failed, Resubmitted}

Transitions:
  Idle      → Running       [execute(fn, data) called]
  Running   → Done          [∀ task ∈ Completed]
  Running   → Cancelled     [error_count ≥ max_errors ∨ SIGINT]
  Running   → Error         [cancel_jobs.is_set after completion]

Per-task:
  Pending     → Executing     [thread picks up task]
  Executing   → Completed     [fn returns non-exception]
  Executing   → Failed        [fn returns exception]
  Executing   → Resubmitted   [elapsed ≥ timeout ∧ remaining ≤ straggler_limit ∧ ¬already_resubmitted]
  Resubmitted → Completed     [resubmitted task completes first]
  Failed      → (error_count incremented; if ≥ max_errors → cancel_jobs.set)
```

---

## 11. TLA+ Specifications

### 11.1. Settings Concurrency

```tla+
---- MODULE SettingsConcurrency ----
EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS MAX_THREADS
ASSUME MAX_THREADS \in 2..3

Threads == 1..MAX_THREADS

VARIABLES
  main_config,            \* Global config: record of key-value
  owner_thread,           \* Thread ID that owns configure, or 0
  thread_overrides,       \* Function: Threads → config overlay
  pc                      \* Program counter: Threads → {"idle","configure","context_enter","context_body","context_exit","read","done"}

vars == <<main_config, owner_thread, thread_overrides, pc>>

TypeOK ==
  /\ main_config \in [{"lm","adapter"} -> {0,1}]
  /\ owner_thread \in 0..MAX_THREADS
  /\ thread_overrides \in [Threads -> [{"lm","adapter"} -> {0,1,2}]]
  /\ pc \in [Threads -> {"idle","configure","context_enter","context_body","context_exit","read","done"}]

Init ==
  /\ main_config = [k \in {"lm","adapter"} |-> 0]
  /\ owner_thread = 0
  /\ thread_overrides = [t \in Threads |-> [k \in {"lm","adapter"} |-> 0]]
  /\ pc = [t \in Threads |-> "idle"]

\* --- Actions ---

Configure(t) ==
  /\ pc[t] = "idle"
  /\ \/ owner_thread = 0
     \/ owner_thread = t
  /\ owner_thread' = t
  /\ main_config' = [k \in {"lm","adapter"} |-> 1]
  /\ pc' = [pc EXCEPT ![t] = "done"]
  /\ UNCHANGED thread_overrides

ConfigureFail(t) ==
  /\ pc[t] = "idle"
  /\ owner_thread /= 0
  /\ owner_thread /= t
  /\ pc' = [pc EXCEPT ![t] = "done"]
  /\ UNCHANGED <<main_config, owner_thread, thread_overrides>>

ContextEnter(t) ==
  /\ pc[t] = "idle"
  /\ thread_overrides' = [thread_overrides EXCEPT ![t] = [k \in {"lm","adapter"} |-> 2]]
  /\ pc' = [pc EXCEPT ![t] = "context_body"]
  /\ UNCHANGED <<main_config, owner_thread>>

ContextExit(t) ==
  /\ pc[t] = "context_body"
  /\ thread_overrides' = [thread_overrides EXCEPT ![t] = [k \in {"lm","adapter"} |-> 0]]
  /\ pc' = [pc EXCEPT ![t] = "done"]
  /\ UNCHANGED <<main_config, owner_thread>>

Read(t) ==
  /\ pc[t] \in {"idle", "context_body"}
  /\ pc' = [pc EXCEPT ![t] = pc[t]]
  /\ UNCHANGED <<main_config, owner_thread, thread_overrides>>

Next ==
  \E t \in Threads:
    \/ Configure(t)
    \/ ConfigureFail(t)
    \/ ContextEnter(t)
    \/ ContextExit(t)
    \/ Read(t)

\* --- Safety ---

\* At most one thread owns configure
SingleOwner == Cardinality({t \in Threads : owner_thread = t}) <= 1

\* Context overrides never mutate global config
ContextIsolation ==
  \A t \in Threads:
    pc[t] = "context_body" =>
      main_config = main_config

\* A non-owner thread never successfully configures
NonOwnerCannotConfigure ==
  \A t \in Threads:
    (owner_thread /= 0 /\ owner_thread /= t) =>
      pc[t] /= "configure"

\* --- Liveness ---

\* Every thread eventually reaches "done" or stays idle
Termination == <>(\A t \in Threads: pc[t] \in {"done", "idle"})

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

====
```

### 11.2. ReAct Agent Loop

```tla+
---- MODULE ReActLoop ----
EXTENDS Naturals, Sequences

CONSTANTS MAX_ITERS, MAX_TRUNCATIONS
ASSUME MAX_ITERS \in 2..5
ASSUME MAX_TRUNCATIONS \in 1..3   \* Python reference: MAX_TRUNCATIONS = 3 (for _ in range(3))

VARIABLES
  iteration,          \* 0..MAX_ITERS
  traj_len,           \* trajectory length (in 4-key groups)
  truncation_attempt, \* 0..MAX_TRUNCATIONS
  tool_result,        \* {"ok", "error", "finish"}
  pc                  \* {"init","call_react","process_tool","truncate_react",
                      \*  "call_extract","truncate_extract","done","error"}

vars == <<iteration, traj_len, truncation_attempt, tool_result, pc>>

TypeOK ==
  /\ iteration \in 0..MAX_ITERS
  /\ traj_len \in 0..(MAX_ITERS * 4)
  /\ truncation_attempt \in 0..MAX_TRUNCATIONS
  /\ tool_result \in {"ok", "error", "finish", "none"}
  /\ pc \in {"init","call_react","process_tool","truncate_react",
             "call_extract","truncate_extract","done","error"}

Init ==
  /\ iteration = 0
  /\ traj_len = 0
  /\ truncation_attempt = 0
  /\ tool_result = "none"
  /\ pc = "init"

\* --- Actions ---

StartLoop ==
  /\ pc = "init"
  /\ pc' = "call_react"
  /\ UNCHANGED <<iteration, traj_len, truncation_attempt, tool_result>>

ReactSuccess ==
  /\ pc = "call_react"
  /\ truncation_attempt' = 0
  /\ \E result \in {"ok", "finish"}:
      /\ tool_result' = result
      /\ traj_len' = traj_len + 4
      /\ IF result = "finish"
         THEN pc' = "call_extract"
         ELSE pc' = "process_tool"
  /\ UNCHANGED iteration

ReactContextExceeded ==
  /\ pc = "call_react"
  /\ truncation_attempt < MAX_TRUNCATIONS
  /\ traj_len >= 4
  /\ pc' = "truncate_react"
  /\ UNCHANGED <<iteration, traj_len, truncation_attempt, tool_result>>

TruncateReact ==
  /\ pc = "truncate_react"
  /\ traj_len' = traj_len - 4
  /\ truncation_attempt' = truncation_attempt + 1
  /\ pc' = "call_react"
  /\ UNCHANGED <<iteration, tool_result>>

TruncateReactFail ==
  /\ pc = "truncate_react"
  /\ traj_len < 4
  /\ pc' = "error"
  /\ UNCHANGED <<iteration, traj_len, truncation_attempt, tool_result>>

ReactExhausted ==
  /\ pc = "call_react"
  /\ truncation_attempt >= MAX_TRUNCATIONS
  /\ pc' = "error"
  /\ UNCHANGED <<iteration, traj_len, truncation_attempt, tool_result>>

ProcessTool ==
  /\ pc = "process_tool"
  /\ iteration' = iteration + 1
  /\ IF iteration' >= MAX_ITERS
     THEN pc' = "call_extract"
     ELSE pc' = "call_react"
  /\ truncation_attempt' = 0
  /\ UNCHANGED <<traj_len, tool_result>>

ReactValueError ==
  /\ pc = "call_react"
  /\ pc' = "call_extract"
  /\ UNCHANGED <<iteration, traj_len, truncation_attempt, tool_result>>

ExtractSuccess ==
  /\ pc = "call_extract"
  /\ pc' = "done"
  /\ UNCHANGED <<iteration, traj_len, truncation_attempt, tool_result>>

ExtractContextExceeded ==
  /\ pc = "call_extract"
  /\ truncation_attempt < MAX_TRUNCATIONS
  /\ traj_len >= 4
  /\ pc' = "truncate_extract"
  /\ UNCHANGED <<iteration, traj_len, truncation_attempt, tool_result>>

TruncateExtract ==
  /\ pc = "truncate_extract"
  /\ traj_len' = traj_len - 4
  /\ truncation_attempt' = truncation_attempt + 1
  /\ pc' = "call_extract"
  /\ UNCHANGED <<iteration, tool_result>>

ExtractExhausted ==
  /\ pc = "call_extract"
  /\ truncation_attempt >= MAX_TRUNCATIONS
  /\ pc' = "error"
  /\ UNCHANGED <<iteration, traj_len, truncation_attempt, tool_result>>

Next ==
  \/ StartLoop
  \/ ReactSuccess
  \/ ReactContextExceeded
  \/ TruncateReact
  \/ TruncateReactFail
  \/ ReactExhausted
  \/ ProcessTool
  \/ ReactValueError
  \/ ExtractSuccess
  \/ ExtractContextExceeded
  \/ TruncateExtract
  \/ ExtractExhausted

\* --- Safety ---

\* iteration never exceeds MAX_ITERS
BoundedIterations == iteration <= MAX_ITERS

\* truncation attempts never exceed MAX_TRUNCATIONS
BoundedTruncations == truncation_attempt <= MAX_TRUNCATIONS

\* trajectory length is non-negative
NonNegativeTrajectory == traj_len >= 0

\* terminal states are absorbing
TerminalAbsorbing ==
  (pc \in {"done", "error"}) => (pc' \in {"done", "error"})

\* --- Liveness ---

\* The agent eventually terminates
AgentTerminates == <>(pc \in {"done", "error"})

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

====
```

### 11.3. Parallel Executor

```tla+
---- MODULE ParallelExecutor ----
EXTENDS Naturals, FiniteSets

CONSTANTS MAX_TASKS, MAX_THREADS, MAX_ERRORS
ASSUME MAX_TASKS \in 2..3
ASSUME MAX_THREADS \in 1..3
ASSUME MAX_ERRORS \in 1..2

Tasks == 1..MAX_TASKS

VARIABLES
  task_state,       \* Tasks → {"pending","executing","completed","failed","resubmitted"}
  error_count,      \* 0..MAX_TASKS
  cancelled,        \* boolean
  results,          \* Tasks → {"none","ok","err"}
  pc                \* {"running","done","cancelled","error"}

vars == <<task_state, error_count, cancelled, results, pc>>

TypeOK ==
  /\ task_state \in [Tasks -> {"pending","executing","completed","failed","resubmitted"}]
  /\ error_count \in 0..MAX_TASKS
  /\ cancelled \in BOOLEAN
  /\ results \in [Tasks -> {"none","ok","err"}]
  /\ pc \in {"running","done","cancelled","error"}

Init ==
  /\ task_state = [t \in Tasks |-> "pending"]
  /\ error_count = 0
  /\ cancelled = FALSE
  /\ results = [t \in Tasks |-> "none"]
  /\ pc = "running"

\* --- Actions ---

StartTask(t) ==
  /\ pc = "running"
  /\ ~cancelled
  /\ task_state[t] = "pending"
  /\ Cardinality({t2 \in Tasks : task_state[t2] = "executing"}) < MAX_THREADS
  /\ task_state' = [task_state EXCEPT ![t] = "executing"]
  /\ UNCHANGED <<error_count, cancelled, results, pc>>

CompleteTask(t) ==
  /\ pc = "running"
  /\ task_state[t] = "executing"
  /\ task_state' = [task_state EXCEPT ![t] = "completed"]
  /\ results' = [results EXCEPT ![t] = "ok"]
  /\ UNCHANGED <<error_count, cancelled, pc>>

FailTask(t) ==
  /\ pc = "running"
  /\ task_state[t] = "executing"
  /\ task_state' = [task_state EXCEPT ![t] = "failed"]
  /\ results' = [results EXCEPT ![t] = "err"]
  /\ error_count' = error_count + 1
  /\ IF error_count' >= MAX_ERRORS
     THEN cancelled' = TRUE
     ELSE cancelled' = cancelled
  /\ UNCHANGED pc

ResubmitTask(t) ==
  /\ pc = "running"
  /\ ~cancelled
  /\ task_state[t] = "executing"
  /\ Cardinality({t2 \in Tasks : task_state[t2] \in {"completed","failed","resubmitted"}}) >= MAX_TASKS - 3
  /\ task_state' = [task_state EXCEPT ![t] = "resubmitted"]
  /\ UNCHANGED <<error_count, cancelled, results, pc>>

ResubmittedComplete(t) ==
  /\ pc = "running"
  /\ task_state[t] = "resubmitted"
  /\ results[t] = "none"
  /\ task_state' = [task_state EXCEPT ![t] = "completed"]
  /\ results' = [results EXCEPT ![t] = "ok"]
  /\ UNCHANGED <<error_count, cancelled, pc>>

CheckDone ==
  /\ pc = "running"
  /\ \A t \in Tasks: results[t] /= "none"
  /\ IF cancelled
     THEN pc' = "error"
     ELSE pc' = "done"
  /\ UNCHANGED <<task_state, error_count, cancelled, results>>

CheckCancelled ==
  /\ pc = "running"
  /\ cancelled
  /\ pc' = "error"
  /\ UNCHANGED <<task_state, error_count, cancelled, results>>

Next ==
  \/ \E t \in Tasks:
       \/ StartTask(t)
       \/ CompleteTask(t)
       \/ FailTask(t)
       \/ ResubmitTask(t)
       \/ ResubmittedComplete(t)
  \/ CheckDone
  \/ CheckCancelled

\* --- Safety ---

\* error_count never exceeds MAX_TASKS
BoundedErrors == error_count <= MAX_TASKS

\* Concurrent executing tasks bounded by thread count
ThreadBound == Cardinality({t \in Tasks : task_state[t] = "executing"}) <= MAX_THREADS

\* A completed result is never overwritten
ResultsMonotonic ==
  \A t \in Tasks:
    results[t] = "ok" => results'[t] = "ok"

\* --- Liveness ---

\* Eventually all tasks resolve
AllTasksResolve == <>(\A t \in Tasks: task_state[t] \in {"completed","failed","resubmitted"})

\* The executor eventually terminates
ExecutorTerminates == <>(pc \in {"done", "error"})

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

====
```

---

## 12. Cross-Cutting Invariants

### 12.1. Safety Invariants (must hold in all states)

```
S1: ∀ sig: sig.input_fields.keys ∩ sig.output_fields.keys == ∅
S2: ∀ predict: predict.lm ≠ null ∨ settings.lm ≠ null    (at call time)
S3: ∀ module: module._compiled ∈ {true, false}
S4: ∀ example: example._input_keys ≠ null → example._input_keys ⊆ example._store.keys
S5: ∀ completions: ∀ k1, k2 ∈ completions.keys: |completions[k1]| == |completions[k2]|
S6: ∀ prediction: ¬hasattr(prediction, _input_keys) ∧ ¬hasattr(prediction, _demos)
S7: ∀ react: "finish" ∈ react.tools.keys
S8: ∀ settings_thread t: (owner_thread ≠ 0 ∧ owner_thread ≠ t) → configure(t) raises RuntimeError
S9: GLOBAL_HISTORY.len ≤ MAX_HISTORY_SIZE (10000)
```

### 12.2. Liveness Guarantees

```
L1: Predict.forward terminates (given LM terminates)
L2: ReAct.forward terminates within max_iters + truncation retries
L3: ParallelExecutor terminates (bounded tasks, bounded errors)
L4: Context manager restores overrides (finally block)
```

### 12.3. Temporal Logic

```
□(S1 ∧ S5 ∧ S6 ∧ S7 ∧ S9)                           -- always hold
□(settings.context_enter → ◇ settings.context_exit)    -- context always exits
□(module.__call__ → ◇ (result ∨ exception))             -- calls terminate
□(react.iteration_start → ◇ react.terminal)            -- agent terminates
```

---

## 13. AX Test Oracle Strategy

The existing TypeScript DSPy port [ax-llm/ax](https://github.com/ax-llm/ax) is used
**exclusively as a test oracle** — never as a code source. AX has battle-tested edge
cases we can cross-validate against, but its architecture (2,700-line monoliths, `as any`
escape hatches, flat 20+ key option bags, Docker-dependent optimizers, void-returning
evaluation) is incompatible with this spec's design contracts.

### 13.1. Oracle Protocol

For each spec fixture file (e.g., `signature_parse.json`, `infer_prefix.json`), the
validation process is:

```
1. Feed equivalent inputs to both our implementation and AX's implementation
2. Compare outputs:
   a. AGREE   → high confidence in correctness
   b. DISAGREE, spec has fixture → spec wins; document the divergence
   c. DISAGREE, no fixture       → investigate; add fixture for the resolved answer
3. AX-only edge cases (inputs our spec doesn't cover) → evaluate for inclusion
```

### 13.2. Oracle Scope by Module

| Spec Section | AX Module | Oracle Value | Reject |
|---|---|---|---|
| §1.4 Signature Parse | `dsp/sig.ts` AxSignature string parser | Cross-validate parse edge cases (nested types, optionals) | Mutable signature with SHA-256 hash; `as any` casts |
| §1.5 infer_prefix | `dsp/sig.ts` field name → prefix | Validate camelCase/snake_case/acronym splits | N/A (pure function, safe to compare) |
| §2 Example/Prediction | `dsp/generate.ts` output handling | Validate completions → prediction construction | `AxGenOut = any` default; untyped fallbacks |
| §3.2 named_parameters | `dsp/program.ts` namedPrograms() | Validate tree walk with nested modules | Static `_propagating` flag; our `visited: Set` is better |
| §4.2 Predict pipeline | `dsp/generate.ts` AxGen.forward() | Validate temperature auto-adjust, LM resolution order | Flat `AxProgramForwardOptions`; 20+ key bag |
| §5.4 ChatAdapter parse | `dsp/extract.ts` | Validate `[[ ## field ## ]]` delimiter extraction | N/A (algorithm comparison only) |
| §8 ReAct | `prompts/agent/` AxAgent | Validate tool dispatch, finish detection | Over-engineered agent loop with selfTuning |
| §10.3 Parallel executor | (not implemented in AX) | No oracle value | AX eval is sequential; our TLA+ spec is authoritative |

### 13.3. What to Study (Reference Only)

These AX techniques are worth understanding as prior art, but must be
**reimplemented from our spec**, not ported:

- **Recursive template-literal type inference** (`dsp/sigtypes.ts`): The technique of
  parsing `"q: string -> a: string"` at the TypeScript type level. Our `TypeTag` universe
  is richer, so the parser needs fresh implementation.
- **Provider capability detection** (`ai/base.ts` `getFeatures()`): Runtime probing for
  structured-output / function-calling support. Useful pattern, but provider integration
  is external to `dspy-slim` core.
- **BootstrapFewShot** (`dsp/optimizers/bootstrapFewshot.ts`): The only AX optimizer
  that runs natively in TS. Reference for Tier 3 optimization work.

### 13.4. Categorical Rejections

The following AX patterns are **explicitly prohibited** in this port:

1. **`as any` private-field mutation** — every interface is a strict contract (§12)
2. **Monolith files** — no file exceeds one abstraction boundary
3. **Docker-dependent optimizers** — all optimizers must run natively in TS
4. **Void-returning evaluation** — `Evaluate` must return typed scores (§10.3)
5. **`IN = any, OUT = any` generic defaults** — all public generics require explicit type parameters
6. **Flat option bags** — pipeline phases use phase-specific typed inputs (§4.2)
7. **`object` in discriminated unions** — `FieldValue` types must be fully discriminated

---

## 14. Port Implementation Checklist

### 14.1. Tier 1 — Required for minimal viable port

```
[ ] Field, Signature, SignatureOps (§1.2–§1.4)
[ ] infer_prefix (§1.5)
[ ] Example, Prediction, Completions (§2)
[ ] BaseModule, Module (§3)
[ ] Parameter (§3.1)
[ ] Predict (§4.1–§4.2)
[ ] ChainOfThought (§4.3)
[ ] Adapter interface (§5.2)
[ ] ChatAdapter parse/format (§5.4, §5.6)
[ ] JSONAdapter parse/format (§5.5)
[ ] BaseLM interface (§6.1)
[ ] LM (OpenAI-compatible client) (§6)
[ ] Settings (configure/context) (§11.1)
```

### 14.2. Tier 2 — Required for agent workflows

```
[ ] Tool, ToolCalls (§7)
[ ] ReAct (§8, §10.2)
[ ] History (§7.3)
[ ] Native function calling adapter path (§5.3)
```

### 14.3. Tier 3 — Required for optimization

```
[ ] Parallel executor (§10.3)
[ ] Callback system (§9)
[ ] GEPA integration (external package)
[ ] Evaluate
```
