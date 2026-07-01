import random
import re
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel


class RawItem(BaseModel):
    question: str
    answer: str
    domain: str
    source: str


class QwenExample(BaseModel):
    messages: list[dict]
    _domain: str = ""


@dataclass
class DomainConfig:
    system_prompts: list[str]
    question_frames: list[str]
    answer_transforms: list[Callable[[str, str], str]]
    keywords: list[str] = field(default_factory=list)


def _extract_topic(text: str) -> tuple[str, str]:
    """Pull a meaningful topic from text — skip stop words."""
    stop = {"what", "how", "why", "when", "where", "who", "is", "are", "the",
            "a", "an", "in", "on", "of", "to", "do", "you", "can", "will",
            "write", "create", "build", "make", "implement", "explain", "describe"}
    words = re.findall(r'\b[A-Za-z][a-zA-Z0-9]+(?:\s+[A-Za-z][a-zA-Z0-9]+)?\b', text)
    meaningful = [w for w in words if w.lower().split()[0] not in stop and len(w) > 3]
    topic = meaningful[0] if meaningful else (words[0] if words else "this concept")
    subtopic = meaningful[1] if len(meaningful) > 1 else "modern systems"
    return topic, subtopic


def _numbered_steps(answer: str) -> str:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if len(s.strip()) > 10]
    if len(sentences) < 2:
        return answer
    steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences[:8]))
    return f"Here's a step-by-step breakdown:\n\n{steps}"


def _explain_then_example(answer: str, topic: str) -> str:
    mid = max(len(answer) // 2, 100)
    concept = answer[:mid].rstrip()
    example = answer[mid:].lstrip() or f"For example, when working with {topic}, practitioners apply these principles directly in real-world scenarios."
    return f"**Concept Overview:**\n{concept}\n\n**Practical Example:**\n{example}"


def _compare_contrast(answer: str, topic: str) -> str:
    return (
        f"**Understanding {topic}:**\n\n"
        f"{answer}\n\n"
        f"**Key Takeaway:** Mastering {topic} requires both theoretical understanding and hands-on practice. "
        f"Always consider the trade-offs between different approaches when applying these concepts."
    )


def _backend_code_transform(ans: str, topic: str) -> str:
    cls_name = topic.replace(' ', '').replace('-', '').replace('.', '')
    route = topic.lower().replace(' ', '-').replace('.', '')
    fn_name = topic.lower().replace(' ', '_').replace('-', '_').replace('.', '_')
    return (
        f"**Backend Design: {topic}**\n\n{ans}\n\n"
        f"```python\n"
        f"# FastAPI implementation example\n"
        f"from fastapi import FastAPI, HTTPException\n"
        f"from pydantic import BaseModel\n\n"
        f"app = FastAPI()\n\n"
        f"class {cls_name}Request(BaseModel):\n"
        f"    pass  # Define request schema fields here\n\n"
        f"@app.post('/{route}')\n"
        f"async def handle_{fn_name}(request: {cls_name}Request):\n"
        f"    try:\n"
        f"        result = await process_{fn_name}(request)\n"
        f"        return {{'status': 'success', 'data': result}}\n"
        f"    except Exception as e:\n"
        f"        raise HTTPException(status_code=500, detail=str(e))\n"
        f"```\n\n"
        f"**Security:** Always validate and sanitize inputs, use parameterized queries, and implement rate limiting."
    )


def _frontend_component_transform(ans: str, topic: str) -> str:
    comp_name = topic.replace(' ', '').replace('-', '').replace('.', '') or "MyComponent"
    fn_name = topic.lower().replace(' ', '_').replace('-', '_').replace('.', '_')
    return (
        f"**Frontend Implementation: {topic}**\n\n{ans}\n\n"
        f"```javascript\n"
        f"// Component implementation\n"
        f"export function {comp_name}Component({{ data, onAction }}) {{\n"
        f"    const [state, setState] = useState(null);\n"
        f"    const [loading, setLoading] = useState(false);\n"
        f"    const [error, setError] = useState(null);\n\n"
        f"    useEffect(() => {{\n"
        f"        // Fetch or compute {topic} data\n"
        f"    }}, [data]);\n\n"
        f"    if (loading) return <LoadingSpinner />;\n"
        f"    if (error) return <ErrorMessage error={{error}} />;\n"
        f"    return <div>{{/* {topic} UI */}}</div>;\n"
        f"}}\n"
        f"```\n\n"
        f"**Accessibility:** Add appropriate ARIA attributes and ensure keyboard navigation works correctly."
    )


# ── Domain Configurations ─────────────────────────────────────────────────────

DOMAIN_CONFIGS: dict[str, DomainConfig] = {
    "cybersecurity": DomainConfig(
        system_prompts=[
            "You are a senior cybersecurity professional with 15 years of experience in penetration testing, threat analysis, and defensive security. You provide detailed, accurate, and ethical guidance on security topics.",
            "You are a cybersecurity expert specializing in red team operations and incident response. You explain complex attack vectors and defenses clearly, always emphasizing ethical and legal considerations.",
            "You are a certified information security professional (CISSP, CEH) who teaches cybersecurity concepts with real-world examples and practical defensive strategies.",
            "You are a threat intelligence analyst who helps teams understand adversary tactics, techniques, and procedures (TTPs) using the MITRE ATT&CK framework.",
            "You are a security architect who designs and explains secure systems, focusing on defense-in-depth strategies and zero-trust principles.",
        ],
        question_frames=[
            "Walk me through how {topic} works and explain why it's a critical concern in modern {subtopic} environments.",
            "What are the practical steps a defender should take to protect against {topic} attacks?",
            "Explain the mechanics of {topic} and describe how an attacker would exploit this vulnerability in practice.",
            "How does {topic} relate to the broader landscape of {subtopic} threats, and what detection strategies are most effective?",
            "Break down the anatomy of a {topic} attack — from initial reconnaissance through exploitation and persistence.",
            "Compare the offensive and defensive perspectives on {topic}: what makes it dangerous and how do we mitigate it?",
            "In a real incident involving {topic}, what should the incident response team prioritize in the first 24 hours?",
            "What tools and techniques are used to detect and analyze {topic} in a corporate network environment?",
        ],
        answer_transforms=[
            lambda ans, topic: f"**Attack Overview:**\n{_explain_then_example(ans, topic)}\n\n**Detection Indicators:**\nMonitor for unusual {topic}-related activity in logs, network traffic, and endpoint telemetry.\n\n**Mitigation Steps:**\n1. Apply relevant patches and security updates\n2. Implement least-privilege access controls\n3. Enable detailed logging and alerting\n4. Conduct regular security assessments",
            lambda ans, topic: _numbered_steps(ans) + f"\n\n**Defense-in-Depth Note:** For {topic}, layer your controls — technical, administrative, and physical measures work together to reduce risk.",
            lambda ans, topic: _compare_contrast(ans, topic) + f"\n\n**MITRE ATT&CK Mapping:** This technique aligns with common adversary TTPs — consult the framework for additional context on {topic} and related sub-techniques.",
        ],
    ),
    "networking": DomainConfig(
        system_prompts=[
            "You are a senior network engineer with deep expertise in TCP/IP, routing protocols, and enterprise network architecture. You explain complex networking concepts clearly with protocol-level detail.",
            "You are a network architect who designs and troubleshoots large-scale networks. You understand both legacy protocols and modern SDN/cloud networking paradigms.",
            "You are a Cisco-certified network professional (CCIE) who teaches networking fundamentals through practical examples and packet-level analysis.",
            "You are a network security specialist who focuses on secure network design, traffic analysis, and protocol vulnerabilities.",
            "You are a telecommunications engineer with expertise in both wired and wireless networking, QoS, and network performance optimization.",
        ],
        question_frames=[
            "What happens at the packet level when {topic} occurs, and which OSI layers are involved?",
            "Explain how {topic} works in a large enterprise network and what role {subtopic} plays in the process.",
            "Describe the protocol exchange involved in {topic} — include the handshake sequence and key fields.",
            "How does {topic} affect network performance, and what configuration changes optimize it?",
            "Walk through a troubleshooting scenario where {topic} is causing connectivity issues — what's your diagnostic approach?",
            "Compare {topic} with alternative protocols or approaches — what are the trade-offs?",
            "How is {topic} implemented differently in IPv4 versus IPv6 environments?",
            "Design a network segment that properly handles {topic} at scale — what are the key considerations?",
        ],
        answer_transforms=[
            lambda ans, topic: f"**Protocol Overview:**\n{ans}\n\n**Protocol Flow:**\n1. Client initiates {topic} process\n2. Protocol negotiation and handshake occur\n3. Data exchange follows established session\n4. Connection teardown or timeout handling\n\n**Wireshark Tip:** Filter with relevant display filters to capture and analyze {topic} traffic in real time.",
            lambda ans, topic: _explain_then_example(ans, topic) + f"\n\n**OSI Layer Reference:** {topic} primarily operates at the relevant OSI layer, interacting with adjacent layers for encapsulation and delivery.",
            lambda ans, topic: _numbered_steps(ans) + f"\n\n**Performance Note:** When optimizing {topic} at scale, consider buffer sizes, timeout values, and QoS classification to maintain throughput under load.",
        ],
    ),
    "it": DomainConfig(
        system_prompts=[
            "You are an experienced IT systems administrator with expertise in Windows, Linux, virtualization, and cloud infrastructure. You provide practical, actionable IT guidance.",
            "You are an IT architect who designs enterprise infrastructure solutions. You have deep knowledge of Active Directory, cloud platforms, and hybrid environments.",
            "You are a DevOps/SRE engineer who bridges IT operations and software development, specializing in automation, monitoring, and reliability engineering.",
            "You are an IT support specialist with broad knowledge of troubleshooting hardware, software, and network issues across diverse enterprise environments.",
            "You are a cloud infrastructure engineer with expertise in AWS, Azure, and GCP, focusing on scalable, cost-effective IT architecture.",
        ],
        question_frames=[
            "How do you configure and manage {topic} in an enterprise environment with hundreds of users?",
            "What's the best practice approach to troubleshooting {topic} when it affects multiple systems simultaneously?",
            "Explain how {topic} integrates with {subtopic} in a hybrid cloud/on-premises IT environment.",
            "Walk through the process of automating {topic} management using scripting or infrastructure-as-code.",
            "What are the key performance metrics to monitor for {topic}, and how do you set up effective alerting?",
            "How does {topic} differ between on-premises and cloud deployments, and what migration considerations matter?",
            "Describe a disaster recovery strategy for systems that rely heavily on {topic}.",
            "What security hardening steps are essential when deploying {topic} in a production environment?",
        ],
        answer_transforms=[
            lambda ans, topic: f"**Overview:**\n{ans}\n\n**Implementation Checklist:**\n- [ ] Document current state before changes\n- [ ] Test in staging/dev environment first\n- [ ] Apply changes during maintenance window\n- [ ] Verify functionality post-change\n- [ ] Update runbooks and documentation",
            lambda ans, topic: _numbered_steps(ans) + f"\n\n**Automation Opportunity:** Consider scripting repetitive {topic} tasks with PowerShell, Bash, or Ansible to reduce manual effort and human error.",
            lambda ans, topic: _explain_then_example(ans, topic) + f"\n\n**Monitoring:** Set up alerts for {topic}-related events using your preferred monitoring stack (Nagios, Datadog, Prometheus, etc.).",
        ],
    ),
    "reasoning": DomainConfig(
        system_prompts=[
            "You are an expert problem solver who breaks down complex logical and mathematical problems into clear, systematic steps. You show your full reasoning process.",
            "You are a mathematics tutor who excels at explaining problem-solving strategies. You always show step-by-step working and explain the reasoning behind each step.",
            "You are an analytical reasoning expert who helps people develop rigorous thinking skills. You emphasize understanding the 'why' behind each logical step.",
            "You are a competitive programmer and mathematician who approaches problems methodically, always verifying solutions and considering edge cases.",
            "You are a critical thinking coach who teaches structured problem-solving frameworks applicable to both technical and real-world challenges.",
        ],
        question_frames=[
            "Work through this {topic} problem step by step, explaining your reasoning at each stage.",
            "What's the most systematic approach to solving problems involving {topic}? Walk me through with an example.",
            "Explain the key insight that makes {topic} problems tractable, then demonstrate with a worked example.",
            "How do you approach {topic} when the problem has multiple possible solution paths? Which do you choose and why?",
            "Break down the logic behind {topic} — what assumptions are being made and how do we verify our answer?",
            "What common mistakes do people make when reasoning about {topic}, and how do you avoid them?",
            "Demonstrate how to apply {topic} concepts to solve a real-world problem from start to finish.",
            "If someone struggled with {topic}, what fundamental concept are they likely missing? Explain it clearly.",
        ],
        answer_transforms=[
            lambda ans, topic: f"**Problem Approach:**\n{ans}\n\n**Step-by-Step Reasoning:**\n{_numbered_steps(ans)}\n\n**Verification:** Always check your answer by working backwards or substituting back into the original problem.",
            lambda ans, topic: f"**Key Insight:**\nThe fundamental concept here is that {topic} requires careful analysis of the given information before jumping to computation.\n\n{ans}\n\n**Common Pitfall:** Many solvers rush through setup — spend extra time ensuring you've correctly interpreted the problem before solving.",
            lambda ans, topic: _explain_then_example(ans, topic) + "\n\n**Practice Tip:** Solving variations of the same problem type builds the pattern recognition needed for speed and accuracy.",
        ],
    ),
    "threejs": DomainConfig(
        system_prompts=[
            "You are an expert Three.js developer with deep knowledge of WebGL, 3D mathematics, and real-time graphics programming. You write clean, performant code with clear explanations.",
            "You are a creative technologist specializing in Three.js and WebGL. You build stunning interactive 3D experiences and explain complex graphics concepts accessibly.",
            "You are a senior frontend engineer who specializes in 3D web graphics using Three.js, with expertise in shaders, geometry, and animation systems.",
            "You are a game developer who brings interactive 3D experiences to the web using Three.js, focusing on performance optimization and visual quality.",
            "You are a WebGL and Three.js instructor who teaches 3D graphics programming from fundamentals through advanced techniques like custom shaders and post-processing.",
        ],
        question_frames=[
            "How do you implement {topic} in Three.js? Show me a complete, working example with explanation.",
            "Build a Three.js scene that demonstrates {topic} — include the geometry setup, materials, lighting, and animation loop.",
            "Explain how {topic} works in the context of WebGL and Three.js internals, then show the implementation.",
            "What's the most performant way to achieve {topic} in Three.js for a scene with thousands of objects?",
            "Walk me through creating a {topic} effect in Three.js, including how to customize it with shader uniforms.",
            "How do you combine {topic} with Three.js post-processing to create a professional visual effect?",
            "Implement {topic} in Three.js and explain how the underlying 3D mathematics make it work.",
            "What are the common pitfalls when implementing {topic} in Three.js, and how do you avoid them?",
        ],
        answer_transforms=[
            lambda ans, topic: f"**Concept:** {topic} in Three.js works by leveraging WebGL's rendering pipeline.\n\n{ans}\n\n```javascript\n// Three.js implementation\nimport * as THREE from 'three';\n\n// Scene setup\nconst scene = new THREE.Scene();\nconst camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);\nconst renderer = new THREE.WebGLRenderer({{ antialias: true }});\nrenderer.setSize(window.innerWidth, window.innerHeight);\ndocument.body.appendChild(renderer.domElement);\n\n// Animation loop\nfunction animate() {{\n    requestAnimationFrame(animate);\n    // Update {topic} logic here\n    renderer.render(scene, camera);\n}}\nanimate();\n```\n\n**Performance Note:** Use `renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))` for sharp rendering without excessive GPU load.",
            lambda ans, topic: _explain_then_example(ans, topic) + f"\n\n```javascript\n// Key {topic} snippet\n// Dispose of geometry and materials when no longer needed\ngeometry.dispose();\nmaterial.dispose();\n```\n\n**Browser Compatibility:** Test in Chrome, Firefox, and Safari — WebGL support is excellent across modern browsers.",
            lambda ans, topic: _numbered_steps(ans) + f"\n\n**Shader Tip:** For advanced {topic} effects, custom GLSL shaders via `THREE.ShaderMaterial` give you full control over the GPU pipeline.",
        ],
    ),
    "animations": DomainConfig(
        system_prompts=[
            "You are a motion design engineer who specializes in web animations using CSS, JavaScript, and Three.js. You create smooth, performant animations that enhance user experience.",
            "You are an animation expert who understands both the artistic principles of motion design and the technical implementation in browsers and WebGL.",
            "You are a frontend developer specializing in high-performance animations using the Web Animations API, GSAP, and CSS animations.",
            "You are a creative developer who builds immersive animated experiences for the web, with expertise in easing functions, spring physics, and procedural animation.",
            "You are a UX engineer focused on purposeful animation — you design motion that guides users, provides feedback, and creates delight without sacrificing performance.",
        ],
        question_frames=[
            "How do you implement a smooth {topic} animation that runs at 60fps without jank?",
            "Explain the principles behind {topic} and show how to implement it using CSS and/or JavaScript.",
            "What's the difference between CSS-based and JavaScript-based approaches to {topic}, and when do you choose each?",
            "Build a {topic} animation sequence that chains multiple effects together with proper timing.",
            "How do you optimize {topic} for mobile devices where GPU and CPU resources are limited?",
            "Walk through creating a {topic} effect using the Web Animations API — include easing and timing controls.",
            "How do you make {topic} accessible for users with vestibular disorders while maintaining the visual effect?",
            "Implement {topic} with smooth enter/exit transitions that respond to user interaction.",
        ],
        answer_transforms=[
            lambda ans, topic: f"**Animation Overview:**\n{ans}\n\n```css\n/* CSS {topic} implementation */\n@keyframes animate {{\n    from {{ opacity: 0; transform: translateY(20px); }}\n    to {{ opacity: 1; transform: translateY(0); }}\n}}\n\n.element {{\n    animation: animate 0.3s ease-out forwards;\n    /* Use transform and opacity for GPU-accelerated animations */\n}}\n\n@media (prefers-reduced-motion: reduce) {{\n    .element {{ animation: none; }}\n}}\n```\n\n**Performance Rule:** Always animate `transform` and `opacity` — they trigger only the composite step, not layout or paint.",
            lambda ans, topic: _explain_then_example(ans, topic) + f"\n\n**Easing Guide:** For {topic}, ease-out feels natural for entering elements, ease-in for exiting, and ease-in-out for looping animations.",
            lambda ans, topic: _numbered_steps(ans) + "\n\n**Accessibility:** Always respect `prefers-reduced-motion` media query — wrap motion-heavy code in a check and provide a static alternative.",
        ],
    ),
    "css": DomainConfig(
        system_prompts=[
            "You are a CSS expert with deep knowledge of the cascade, specificity, layout algorithms, and modern CSS features. You write maintainable, scalable stylesheets.",
            "You are a frontend engineer specializing in CSS architecture. You design component-based styling systems using modern CSS features and methodologies.",
            "You are a web designer and CSS specialist who creates beautiful, responsive layouts using Grid, Flexbox, and custom properties.",
            "You are a CSS performance expert who optimizes rendering pipelines, reduces layout thrashing, and implements efficient selectors at scale.",
            "You are a design systems engineer who builds robust CSS token systems and utility-first frameworks for large frontend teams.",
        ],
        question_frames=[
            "Explain how {topic} works in CSS and show a practical implementation with common use cases.",
            "Design a responsive layout using {topic} that works across mobile, tablet, and desktop breakpoints.",
            "What are the gotchas and edge cases with {topic} in CSS, and how do you handle them reliably?",
            "Show me how to implement {topic} using modern CSS without JavaScript — include browser compatibility notes.",
            "Compare {topic} with the older approach to the same problem — why is the modern approach better?",
            "How do you use CSS custom properties (variables) to make {topic} themeable and maintainable?",
            "Build a component that demonstrates {topic} with clean, reusable CSS that follows BEM or similar methodology.",
            "How does {topic} interact with the CSS cascade and specificity? Show examples of where it might surprise you.",
        ],
        answer_transforms=[
            lambda ans, topic: f"**CSS Concept: {topic}**\n\n{ans}\n\n```css\n/* Modern CSS implementation */\n.container {{\n    /* {topic} applied here */\n    container-type: inline-size;\n}}\n\n/* Responsive variant */\n@container (min-width: 600px) {{\n    .element {{\n        /* Adjusted styles */\n    }}\n}}\n```\n\n**Browser Support:** Check Can I Use for `{topic}` — modern browsers have excellent support, but verify for your target audience.",
            lambda ans, topic: _explain_then_example(ans, topic) + f"\n\n**Specificity Note:** When using {topic}, be mindful of the cascade — lower-specificity utility classes may be overridden unexpectedly.",
            lambda ans, topic: _numbered_steps(ans) + f"\n\n**Debugging Tip:** Use browser DevTools' computed styles panel to inspect how {topic} is being applied and where inheritance or specificity conflicts arise.",
        ],
    ),
    "frontend": DomainConfig(
        system_prompts=[
            "You are a senior frontend engineer with expertise in React, Vue, TypeScript, and modern build tooling. You build performant, accessible, and maintainable user interfaces.",
            "You are a frontend architect who designs scalable component systems and state management solutions for large applications.",
            "You are a web performance specialist who optimizes Core Web Vitals, bundle sizes, and rendering performance in production frontend applications.",
            "You are a UI engineer focused on accessibility (WCAG compliance), semantic HTML, and progressive enhancement.",
            "You are a full-stack developer with deep frontend expertise, specializing in API integration, real-time updates, and modern JavaScript patterns.",
        ],
        question_frames=[
            "How do you implement {topic} in a modern frontend application while maintaining good performance and accessibility?",
            "Design a reusable component that handles {topic} — what's the API design and how do you make it flexible?",
            "Explain the state management approach for {topic} and show how to handle loading, error, and success states.",
            "What are the performance implications of {topic} and how do you optimize it for production?",
            "How do you test {topic} in a frontend application — what test types and what are you verifying?",
            "Walk through implementing {topic} with proper TypeScript typing and error boundary handling.",
            "How does {topic} differ between a single-page app and a server-side rendered application?",
            "What accessibility considerations apply to {topic} and how do you ensure WCAG compliance?",
        ],
        answer_transforms=[
            _frontend_component_transform,
            lambda ans, topic: _explain_then_example(ans, topic) + f"\n\n**Performance:** Memoize expensive computations related to {topic} using `useMemo` or `useCallback` to prevent unnecessary re-renders.",
            lambda ans, topic: _numbered_steps(ans) + f"\n\n**Testing Strategy:** Write unit tests for {topic} business logic, integration tests for API interactions, and E2E tests for critical user flows.",
        ],
    ),
    "backend": DomainConfig(
        system_prompts=[
            "You are a senior backend engineer with expertise in Python, Node.js, databases, and distributed systems. You design scalable, reliable, and secure APIs.",
            "You are a system architect who designs microservices, event-driven systems, and cloud-native backend solutions.",
            "You are a database engineer with deep expertise in SQL, NoSQL, query optimization, and data modeling for high-traffic applications.",
            "You are a backend security specialist who builds authentication systems, validates inputs rigorously, and designs APIs that resist common attack vectors.",
            "You are an API design expert who creates clean RESTful and GraphQL APIs with proper versioning, documentation, and error handling.",
        ],
        question_frames=[
            "Design a backend service that handles {topic} at scale — what's the architecture and key design decisions?",
            "How do you implement {topic} in a REST API with proper error handling, validation, and security?",
            "Explain the database schema design for {topic} and how you'd query it efficiently at high volume.",
            "What caching strategy works best for {topic}, and how do you handle cache invalidation?",
            "How do you implement {topic} with proper authentication and authorization controls?",
            "Walk through the error handling strategy for {topic} — what can fail and how do you handle each case gracefully?",
            "How do you test {topic} in a backend service — unit, integration, and load testing approaches?",
            "Design the API contract for {topic} — endpoint design, request/response schemas, and versioning strategy.",
        ],
        answer_transforms=[
            _backend_code_transform,
            lambda ans, topic: _explain_then_example(ans, topic) + f"\n\n**Scalability:** For {topic} at scale, consider horizontal scaling, connection pooling, and async I/O to handle concurrent requests efficiently.",
            lambda ans, topic: _numbered_steps(ans) + f"\n\n**Observability:** Add structured logging, metrics, and distributed tracing to {topic} endpoints for production debugging.",
        ],
    ),
}

# ── Verbatim Guard ────────────────────────────────────────────────────────────

def _token_overlap(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _shares_ngram(a: str, b: str, n: int = 4) -> bool:
    wa = a.lower().split()
    wb = b.lower().split()
    ngrams_a = {tuple(wa[i:i+n]) for i in range(len(wa) - n + 1)}
    ngrams_b = {tuple(wb[i:i+n]) for i in range(len(wb) - n + 1)}
    return bool(ngrams_a & ngrams_b)


def _passes_verbatim_guard(generated_q: str, generated_a: str, source_q: str, source_a: str) -> bool:
    # Block if the generated question is too close to source question (4-gram match)
    if _shares_ngram(generated_q, source_q, 4):
        return False
    # Block only if generated answer is basically identical to source (exact or near-exact copy)
    if generated_a.strip() == source_a.strip():
        return False
    # Require the generated answer to meaningfully expand on the source (at least 20% longer)
    if len(generated_a) < len(source_a) * 1.2 and _token_overlap(generated_a, source_a) > 0.90:
        return False
    return True


# ── Main Generation Function ──────────────────────────────────────────────────

def build_messages(system: str, user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def generate_example(raw: RawItem) -> dict | None:
    """Transform a RawItem into a Qwen-formatted example. Returns None if verbatim guard fails all attempts."""
    domain_key = raw.domain.lower().replace(" ", "").replace(".", "")
    # Map domain aliases
    alias_map = {
        "cybersec": "cybersecurity", "security": "cybersecurity",
        "network": "networking", "infosec": "cybersecurity",
        "javascript": "threejs", "js": "threejs", "webgl": "threejs",
        "animation": "animations", "motion": "animations",
        "stylesheet": "css", "styling": "css",
        "ui": "frontend", "react": "frontend", "vue": "frontend",
        "api": "backend", "server": "backend", "database": "backend",
        "math": "reasoning", "logic": "reasoning", "problem": "reasoning",
        "sysadmin": "it", "infrastructure": "it", "devops": "it",
    }
    domain_key = alias_map.get(domain_key, domain_key)
    if domain_key not in DOMAIN_CONFIGS:
        # Try partial match
        for k in DOMAIN_CONFIGS:
            if k in domain_key or domain_key in k:
                domain_key = k
                break
        else:
            domain_key = random.choice(list(DOMAIN_CONFIGS.keys()))

    cfg = DOMAIN_CONFIGS[domain_key]
    topic, subtopic = _extract_topic(raw.question)

    frames = cfg.question_frames.copy()
    transforms = cfg.answer_transforms.copy()
    random.shuffle(frames)
    random.shuffle(transforms)

    for attempt in range(3):
        frame = frames[attempt % len(frames)]
        transform = transforms[attempt % len(transforms)]
        system_prompt = random.choice(cfg.system_prompts)

        try:
            new_question = frame.format(topic=topic, subtopic=subtopic)
        except KeyError:
            new_question = frame.replace("{topic}", topic).replace("{subtopic}", subtopic)

        try:
            new_answer = transform(raw.answer, topic)
        except Exception:
            new_answer = _explain_then_example(raw.answer, topic)

        if _passes_verbatim_guard(new_question, new_answer, raw.question, raw.answer):
            example = build_messages(system_prompt, new_question, new_answer)
            example["_domain"] = domain_key
            return example

    return None
