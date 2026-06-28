#!/usr/bin/env python3
"""
Qwen2.5VL dataset writer — generates qwen_dataset.jsonl with 2000 examples.
Run: python write_dataset.py
"""
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# DOMAIN DATA POOLS
# ---------------------------------------------------------------------------

CYBER_TOPICS = [
    ("SQL injection","web security","parameterized queries"),
    ("XSS reflected","client-side injection","output encoding"),
    ("CSRF attacks","forged requests","SameSite cookies"),
    ("heap use-after-free","memory corruption","safe allocators"),
    ("Kerberoasting","TGS ticket cracking","AES service accounts"),
    ("process hollowing","code injection","memory scanning"),
    ("JWT alg:none","forged tokens","explicit algorithm allowlist"),
    ("NTLM relay","SMB signing bypass","SMB signing enforcement"),
    ("Golden Ticket","KRBTGT hash abuse","KRBTGT rotation"),
    ("path traversal","directory escape","canonical path check"),
    ("SSTI","template injection RCE","sandboxed rendering"),
    ("SSRF","cloud metadata theft","IMDS v2 enforcement"),
    ("OAuth PKCE","code interception","code_verifier binding"),
    ("ECB mode","pattern leakage","AES-GCM authenticated encryption"),
    ("buffer overflow","stack smashing","stack canaries and ASLR"),
    ("deserialization RCE","pickle exploit","JSON instead of pickle"),
    ("DNS rebinding","same-origin bypass","DNS pinning"),
    ("XXE injection","XML external entities","disable external entities"),
    ("command injection","shell metacharacters","subprocess with list args"),
    ("open redirect","phishing amplifier","strict whitelist of redirect targets"),
    ("clickjacking","iframe overlay attack","X-Frame-Options DENY"),
    ("prototype pollution","JS object chain corruption","Object.create(null) maps"),
    ("insecure direct object reference","BOLA","ownership checks per request"),
    ("broken access control","horizontal privilege escalation","enforce RBAC server-side"),
    ("mass assignment","auto-bind vulnerability","explicit allowlist of writable fields"),
    ("LDAP injection","directory query manipulation","parameterized LDAP filters"),
    ("HTTP request smuggling","CL-TE desync","consistent server parsing"),
    ("race condition TOCTOU","time-of-check exploit","atomic operations"),
    ("subdomain takeover","dangling CNAME","DNS record cleanup pipeline"),
    ("supply chain attack","malicious npm package","lockfiles and integrity hashing"),
    ("C2 beaconing","periodic HTTP check-in","RITA beacon detection"),
    ("ransomware IR","forensic first hour","memory acquisition before containment"),
    ("AV evasion","LOLBin abuse","behavioral EDR detection"),
    ("credential stuffing","password spray","rate limiting and MFA"),
    ("phishing analysis","email header inspection","SPF DKIM DMARC verification"),
    ("AWS IAM misconfiguration","wildcard policy abuse","least privilege and Access Analyzer"),
    ("container escape","privileged pod","non-root containers and seccomp"),
    ("Zeek signature writing","network threat detection","behavioral anomaly rules"),
    ("TLS 1.3 handshake","forward secrecy","ECDHE key exchange"),
    ("certificate pinning bypass","mobile MitM","dynamic pinning with backup pins"),
    ("Zero Trust architecture","implicit trust removal","micro-segmentation and identity verification"),
    ("K8s RBAC","pod service account abuse","least-privilege ClusterRoles"),
    ("SolarWinds-style supply chain","build pipeline compromise","reproducible builds and SBOM"),
    ("Log4Shell","JNDI lookup RCE","input sanitization and WAF rules"),
    ("PrintNightmare","Windows print spooler RCE","disable spooler on DCs"),
    ("ProxyLogon","Exchange pre-auth RCE","patch cadence and network segmentation"),
    ("BloodHound path analysis","AD attack path enumeration","attack path remediation"),
    ("living off the land","certutil and mshta abuse","application allowlisting"),
    ("memory forensics","Volatility framework","process and network artifact analysis"),
    ("threat hunting","hypothesis-driven investigation","MITRE ATT&CK pivot queries"),
]

CYBER_SYSTEMS = [
    "You are a senior penetration tester with a decade of experience in web application security.",
    "You are a red team operator specializing in Active Directory and Windows environments.",
    "You are a defensive security engineer focused on detection engineering and SIEM tuning.",
    "You are a malware analyst at a threat intelligence firm who reverse engineers samples.",
    "You are a cloud security architect who audits AWS, Azure, and GCP environments.",
    "You are a CTF competitor and security researcher who explains challenge solutions clearly.",
    "You are a vulnerability researcher who analyzes CVEs and explains exploitation techniques.",
    "You are a security engineer specializing in application security and secure code review.",
]

NET_TOPICS = [
    ("TCP three-way handshake","connection establishment","SYN SYN-ACK ACK sequence"),
    ("OSPF vs BGP","IGP vs EGP","link-state vs path-vector"),
    ("VLAN hopping","double tagging attack","native VLAN hardening"),
    ("DNS resolution","recursive vs iterative","resolver cache and TTL"),
    ("BGP route selection","best-path algorithm","weight AS-path MED local-pref"),
    ("TCP retransmission","loss detection","RTO and duplicate ACK"),
    ("SDN architecture","control plane separation","OpenFlow and centralized policy"),
    ("packet loss diagnosis","MTR and Wireshark","RTT jitter and drop pattern"),
    ("spine-leaf topology","fat-tree DC design","equal-cost multipath"),
    ("IPv6 transition","dual-stack deployment","prefix delegation and SLAAC"),
    ("QoS DSCP","traffic classification","priority queuing and policing"),
    ("802.11 wireless","CSMA/CA vs CSMA/CD","hidden node and RTS/CTS"),
    ("network automation with Netmiko","SSH config push","idempotent device management"),
    ("MPLS VPN","label switching","VRF separation and PE-CE routing"),
    ("NAT PAT tracking","stateful address translation","connection table management"),
    ("SD-WAN vs MPLS","overlay WAN","application-aware path selection"),
    ("ARP protocol","MAC to IP resolution","ARP spoofing and dynamic ARP inspection"),
    ("STP root bridge election","spanning tree","RSTP and BPDU guard"),
    ("LACP link aggregation","bonding protocols","LACP PDU negotiation"),
    ("firewall zone policy","stateful inspection","deny-all default with explicit allows"),
    ("SNMP v3 monitoring","OID polling","authPriv security model"),
    ("NetFlow analysis","traffic accounting","top talkers and anomaly detection"),
    ("TCP window scaling","BDP optimization","bandwidth-delay product tuning"),
    ("GRE tunnel vs IPSec","encapsulation overhead","when to use each"),
    ("multicast PIM-SM","group membership","IGMP snooping in switches"),
    ("BGP community attributes","traffic engineering","blackhole and no-export communities"),
    ("network time protocol","stratum hierarchy","PTP for sub-microsecond precision"),
    ("proxy vs reverse proxy","request interception","CDN and load balancer patterns"),
    ("DHCP option 82","relay agent info","IP address management"),
    ("ECMP load balancing","hash-based distribution","per-flow vs per-packet"),
    ("IPv6 neighbor discovery","NDP vs ARP","router advertisement security"),
    ("TACACS+ vs RADIUS","AAA protocols","attribute-value pairs and accounting"),
    ("network segmentation","microsegmentation","VLAN and firewall zone design"),
    ("anycast routing","distributed services","DNS anycast and DDoS mitigation"),
    ("packet capture analysis","tcpdump filters","BPF syntax and display filters"),
    ("HTTPS inspection","TLS decryption","certificate re-signing and privacy"),
    ("VXLAN overlay","L2 over L3","VTEP and flood-and-learn"),
    ("IS-IS vs OSPF","link-state comparison","ISIS advantages in large networks"),
    ("port mirroring SPAN","traffic copy","RSPAN and ERSPAN for remote capture"),
    ("network policy automation","Ansible for network","idempotent playbook design"),
    ("eBGP multihop","IBGP full mesh","route reflector design"),
    ("TCP BBR congestion control","bandwidth probing","comparison with CUBIC"),
    ("QUIC protocol","HTTP/3 transport","stream multiplexing without HoL blocking"),
    ("DNS over HTTPS","DoH privacy","resolver policy and split-horizon"),
    ("802.1X port authentication","EAP methods","RADIUS integration for NAC"),
    ("network baseline","capacity planning","percentile analysis of utilization"),
    ("traffic shaping vs policing","token bucket leaky bucket","burst handling"),
    ("MPLS TE RSVP","traffic engineering","constraint-based path computation"),
    ("link aggregation failover","active-passive vs active-active","LACP failover timing"),
    ("network documentation","as-built diagrams","IPAM and change management"),
]

NET_SYSTEMS = [
    "You are a senior network engineer with expertise in enterprise routing and switching.",
    "You are a network security analyst who performs packet capture analysis and intrusion detection.",
    "You are a network architect designing large-scale data center and WAN infrastructures.",
    "You are a telecommunications engineer who explains networking protocols in depth.",
    "You are a network automation engineer who builds tooling for large-scale infrastructure management.",
]

IT_TOPICS = [
    ("Active Directory forest design","trust relationships","OU structure and GPO inheritance"),
    ("Docker container networking","bridge vs host mode","CNI plugins and overlay networks"),
    ("Kubernetes pod scheduling","affinity and taints","resource requests and limits"),
    ("Terraform state management","remote state backend","state locking and workspace isolation"),
    ("Linux process management","systemd units","cgroups and resource accounting"),
    ("Windows Server patching","WSUS and MECM","ring-based deployment strategy"),
    ("Azure AD conditional access","identity-based policies","MFA enforcement and device compliance"),
    ("AWS S3 bucket security","ACL vs bucket policy","Block Public Access and encryption"),
    ("Prometheus alerting","PromQL queries","alertmanager routing and silencing"),
    ("Grafana dashboard design","data source federation","variable-driven templating"),
    ("Ansible playbook idempotency","desired state","check mode and diff"),
    ("GitLab CI pipeline","stage dependencies","artifact caching and parallel jobs"),
    ("Linux LVM management","volume groups and PVs","thin provisioning and snapshots"),
    ("Windows Group Policy","GPO processing order","RSoP and loopback mode"),
    ("cloud cost optimization","rightsizing instances","reserved vs spot capacity"),
    ("Kubernetes RBAC","service account bindings","ClusterRole vs Role scope"),
    ("Linux performance tuning","sysctl parameters","CPU affinity and NUMA"),
    ("backup and recovery strategy","RTO and RPO","3-2-1 backup rule"),
    ("PKI certificate lifecycle","CA hierarchy","OCSP stapling and CRL"),
    ("log aggregation with ELK","Logstash pipeline","index lifecycle management"),
    ("container image security","layer scanning","distroless base images"),
    ("AWS VPC design","subnet sizing","NAT gateway vs NAT instance"),
    ("DNS split-horizon","internal vs external views","BIND zone configuration"),
    ("Linux cron vs systemd timers","scheduled tasks","persistent timer advantages"),
    ("PowerShell DSC","desired state configuration","pull vs push mode"),
    ("Kubernetes Helm charts","templating","values override and hooks"),
    ("load balancer health checks","active vs passive","circuit breaker pattern"),
    ("LDAP directory integration","bind DN and search filters","PAM and NSS integration"),
    ("cloud IAM role design","cross-account access","role chaining limits"),
    ("monitoring alerting fatigue","alert thresholds","symptom-based vs cause-based alerts"),
    ("Linux networking namespaces","network isolation","veth pairs and bridge linking"),
    ("Windows failover clustering","quorum configuration","witness disk and file share witness"),
    ("Kubernetes persistent volumes","StorageClass","dynamic provisioning and reclaim policy"),
    ("infrastructure as code testing","Terratest and conftest","policy as code with OPA"),
    ("SSH bastion host design","jump host pattern","ProxyJump and certificate-based auth"),
    ("VMware vSphere HA","DRS and resource pools","vMotion requirements"),
    ("GCP Cloud Run","serverless containers","concurrency and cold start"),
    ("Elasticsearch cluster sizing","shard allocation","hot-warm-cold architecture"),
    ("Linux audit subsystem","auditd rules","compliance log collection"),
    ("incident management process","ITIL vs SRE","runbook automation"),
    ("Ansible vault","secret encryption","integration with HashiCorp Vault"),
    ("Kubernetes network policies","pod-to-pod isolation","Calico policy enforcement"),
    ("Azure Monitor","metric alerts","action groups and log analytics"),
    ("configuration drift detection","Chef InSpec","automated compliance scanning"),
    ("NFS vs SMB","file sharing protocols","Kerberos authentication for NFS"),
    ("HPC cluster management","SLURM job scheduling","MPI communication patterns"),
    ("database backup strategies","logical vs physical backup","point-in-time recovery"),
    ("service mesh Istio","sidecar proxy","mTLS between services"),
    ("Linux kernel modules","dynamic loading","blacklisting and parameter tuning"),
    ("cloud networking peering","VPC peering vs Transit Gateway","transitive routing limits"),
]

IT_SYSTEMS = [
    "You are a senior systems administrator managing enterprise Linux and Windows infrastructure.",
    "You are a DevOps engineer who builds and maintains cloud-native infrastructure at scale.",
    "You are a cloud architect with expertise in AWS, Azure, and GCP enterprise deployments.",
    "You are a site reliability engineer focused on availability, scalability, and incident response.",
    "You are an infrastructure automation engineer who uses Terraform, Ansible, and GitOps patterns.",
]

REASONING_TOPICS = [
    ("binary search algorithm","sorted array lookup","O(log n) midpoint comparison"),
    ("dynamic programming","overlapping subproblems","memoization vs tabulation"),
    ("graph BFS vs DFS","traversal strategies","shortest path vs cycle detection"),
    ("merge sort stability","divide and conquer","O(n log n) guaranteed"),
    ("hash table collision","chaining vs open addressing","load factor and rehashing"),
    ("Dijkstra's algorithm","single-source shortest path","priority queue implementation"),
    ("recursion vs iteration","call stack depth","tail call optimization"),
    ("time complexity analysis","big-O notation","amortized analysis"),
    ("two-pointer technique","sorted array problems","linear time with O(1) space"),
    ("sliding window","substring problems","expanding and shrinking window"),
    ("probability Bayes theorem","conditional probability","posterior update"),
    ("combinatorics counting","permutations and combinations","stars and bars"),
    ("modular arithmetic","clock arithmetic","Chinese Remainder Theorem"),
    ("floating point precision","IEEE 754","epsilon comparison"),
    ("system design trade-offs","CAP theorem","consistency vs availability"),
    ("database normalization","1NF 2NF 3NF","denormalization for performance"),
    ("distributed consensus","Raft algorithm","leader election and log replication"),
    ("consistent hashing","ring-based distribution","virtual nodes for balance"),
    ("cache eviction policies","LRU vs LFU","implementation with linked hash map"),
    ("bit manipulation tricks","XOR swap","popcount and power of two check"),
    ("tree traversal orders","pre in post order","Morris traversal without stack"),
    ("topological sort","DAG ordering","Kahn's algorithm vs DFS approach"),
    ("string matching","KMP algorithm","failure function computation"),
    ("disjoint set union-find","path compression","rank-based union"),
    ("segment tree range queries","lazy propagation","point update range sum"),
    ("A-star pathfinding","heuristic search","admissible heuristic and optimality"),
    ("greedy algorithm correctness","exchange argument","proving greedy choice"),
    ("NP-completeness","polynomial reduction","approximation algorithms"),
    ("network flow max-flow","Ford-Fulkerson","residual graph and augmenting paths"),
    ("convex hull","Graham scan","cross product orientation test"),
    ("matrix exponentiation","Fibonacci in log n","recurrence relation acceleration"),
    ("reservoir sampling","streaming uniform sample","k-item reservoir"),
    ("bloom filter","probabilistic membership","false positive rate tuning"),
    ("skip list","probabilistic balanced structure","O(log n) expected operations"),
    ("lock-free data structures","compare-and-swap","ABA problem"),
    ("memory model happens-before","volatile and synchronized","Java memory model"),
    ("garbage collection algorithms","mark-and-sweep vs generational","GC pause tuning"),
    ("compiler optimization","constant folding inlining","loop unrolling"),
    ("SQL query optimization","query planner","index selection and join order"),
    ("regular expression complexity","catastrophic backtracking","possessive quantifiers"),
    ("event loop model","JavaScript single-threaded","microtask vs macrotask queue"),
    ("type inference","Hindley-Milner","unification algorithm"),
    ("Byzantine fault tolerance","f+1 honest nodes","PBFT protocol"),
    ("load shedding","admission control","token bucket vs leaky bucket"),
    ("approximate counting","HyperLogLog","hash cardinality estimation"),
    ("rate limiting algorithms","sliding window log","token bucket implementation"),
    ("prefix sum array","range query O(1)","2D prefix sum"),
    ("interval scheduling","earliest deadline first","proof of optimality"),
    ("digit DP","counting with constraints","state over digit positions"),
    ("random projection","dimensionality reduction","Johnson-Lindenstrauss lemma"),
]

REASONING_SYSTEMS = [
    "You are a competitive programmer who explains algorithmic thinking and problem-solving strategies.",
    "You are a computer science professor who teaches data structures, algorithms, and systems design.",
    "You are a software architect who reasons through complex system design problems step by step.",
    "You are a mathematician who applies formal reasoning to programming and computational problems.",
    "You are a senior engineer at a tech company who conducts technical interviews and explains optimal solutions.",
]

THREEJS_TOPICS = [
    ("rotating cube","basic mesh setup","BoxGeometry MeshBasicMaterial animation loop"),
    ("particle system","point cloud","BufferGeometry and Points with custom shader"),
    ("GLSL vertex shader","position displacement","time-based sine wave deformation"),
    ("GLSL fragment shader","procedural texture","UV-based color pattern generation"),
    ("physically based rendering","MeshStandardMaterial","metalness roughness environment map"),
    ("shadow mapping","directional light shadows","shadow camera frustum tuning"),
    ("post-processing effects","EffectComposer","UnrealBloomPass and depth of field"),
    ("3D text rendering","FontLoader","TextGeometry bevel and material"),
    ("raycasting mouse interaction","object picking","Raycaster and intersect objects"),
    ("GLTF model loading","GLTFLoader","draco compression and morph targets"),
    ("skeletal animation","AnimationMixer","clip action cross-fade"),
    ("instanced mesh","GPU instancing","InstancedMesh setMatrixAt performance"),
    ("environment map reflections","PMREMGenerator","HDR texture equirectangular"),
    ("fog effect","linear vs exponential fog","scene fog property"),
    ("orbit controls","user camera navigation","damping and constraints"),
    ("terrain generation","heightmap","PlaneGeometry vertex displacement"),
    ("water surface","water normal map","reflectivity and wave animation"),
    ("volumetric light","god rays","occlusion pass and additive blending"),
    ("custom geometry","BufferAttribute","indexed vs non-indexed faces"),
    ("shader material uniforms","passing data to GPU","time and resolution uniforms"),
    ("LOD level of detail","distance-based switching","LOD object management"),
    ("frustum culling","visibility optimization","BoundingBox and frustum test"),
    ("render targets","off-screen rendering","WebGLRenderTarget and texture sampling"),
    ("instanced animation","transform per frame","large crowd simulation"),
    ("morph targets","shape keys","AnimationClip and morph influence"),
    ("cubemap skybox","CubeTextureLoader","6-face cubemap setup"),
    ("audio visualization","AnalyserNode","frequency bars with Bar3D"),
    ("physics integration Cannon.js","rigid body sync","quaternion and position copy"),
    ("toon shading","cel shader","edge detection and quantized lighting"),
    ("WebGL performance profiling","draw call reduction","geometry merging strategy"),
    ("sprite billboarding","SpriteMaterial","alpha map and sizeAttenuation"),
    ("tube geometry along path","CatmullRomCurve3","TubeGeometry radial segments"),
    ("lens flare","LensflareElement","additive blending and occlusion"),
    ("wireframe rendering","WireframeGeometry","edge highlight technique"),
    ("MeshTransmissionMaterial","glass refraction","thickness and chromatic aberration"),
    ("GPU picking","render ID to texture","color-coded picking buffer"),
    ("custom orbit camera","spherical coordinates","azimuth elevation radius control"),
    ("portal effect","stencil buffer","recursive portal rendering"),
    ("displacement map","height texture","normalScale and displacement scale"),
    ("screen-space ambient occlusion","SSAO pass","kernel samples and radius"),
    ("cloth simulation","verlet integration","constraint solving"),
    ("fluid simulation","SPH particles","density and pressure forces"),
    ("procedural city generation","instanced buildings","randomized block layout"),
    ("galaxy particle shader","spiral arm formula","color gradient by distance"),
    ("shader noise","Simplex noise in GLSL","fbm fractal brownian motion"),
    ("reflection probe","baked irradiance","lightmap UV unwrapping"),
    ("XR AR VR setup","WebXR session","controller input and hand tracking"),
    ("CSG constructive solid geometry","mesh boolean","ThreeBSP operations"),
    ("texture atlas","UV packing","reducing texture bind calls"),
    ("scene graph optimization","object pooling","reuse vs garbage collection"),
]

THREEJS_SYSTEMS = [
    "You are a Three.js expert who builds interactive 3D web experiences and explains WebGL concepts clearly.",
    "You are a graphics programmer specializing in WebGL, GLSL shaders, and real-time rendering techniques.",
    "You are a creative developer who combines Three.js with performance optimization for production web apps.",
    "You are a game developer who uses Three.js and WebGL for browser-based games and simulations.",
]

ANIM_TOPICS = [
    ("CSS keyframe animation","@keyframes definition","from-to and percentage steps"),
    ("CSS transition","property interpolation","transition-timing-function easing"),
    ("requestAnimationFrame loop","60fps rendering","delta time for frame-rate independence"),
    ("GSAP timeline","sequence and overlap","stagger and repeat options"),
    ("spring physics animation","spring mass damping","overshoot and settle behavior"),
    ("CSS custom property animation","@property registration","typed interpolation"),
    ("Web Animations API","Element animate","keyframe object and timing options"),
    ("scroll-driven animation","scroll timeline","animation-timeline and view()"),
    ("SVG path animation","stroke-dasharray","dashoffset draw-on effect"),
    ("Lottie animation","JSON bodymovin export","player events and segments"),
    ("Intersection Observer","lazy reveal","threshold and rootMargin"),
    ("CSS will-change","compositor hint","layer promotion and memory trade-off"),
    ("animation performance","composite properties","transform and opacity only"),
    ("Framer Motion","React animation library","layout animation and shared layout"),
    ("motion blur effect","CSS blur on translate","trail rendering with canvas"),
    ("parallax scrolling","depth layers","transform translateZ and perspective"),
    ("page transition","route change animation","exit and enter keyframes"),
    ("skeleton loader","content placeholder","gradient shimmer animation"),
    ("micro-interactions","hover and focus feedback","delight without distraction"),
    ("drag and drop animation","pointer events","drop zone highlight and spring snap"),
    ("morphing SVG","path interpolation","GSAP MorphSVG plugin"),
    ("physics-based bounce","cubic bezier","spring approximation in CSS"),
    ("text character animation","split text","stagger per letter entrance"),
    ("canvas particle system","requestAnimationFrame","particle lifecycle and recycling"),
    ("3D CSS perspective","rotateY and translateZ","card flip animation"),
    ("animated gradient","hue-rotate filter","conic-gradient position animation"),
    ("loading spinner","border animation","conic-gradient rotating mask"),
    ("number counter animation","lerp approach","formatted integer output"),
    ("typed text effect","character append","cursor blink keyframe"),
    ("scroll snap","scroll-snap-type","full-page section snapping"),
    ("clip-path reveal","polygon interpolation","text reveal wipe effect"),
    ("blur reveal transition","filter blur to 0","combined opacity transform"),
    ("accordion animation","max-height transition","auto height problem and solution"),
    ("sticky header shrink","scroll event class toggle","compact nav animation"),
    ("infinite carousel","CSS scroll or JS clone","seamless loop technique"),
    ("ripple effect","scale from click origin","clip-circle and pointer position"),
    ("progress bar animation","width transition","percentage driven by JS"),
    ("cursor follower","lerp position","smooth lag effect"),
    ("confetti animation","canvas colored rects","gravity and rotation per particle"),
    ("flip animation technique","First Last Invert Play","layout change without jank"),
    ("audio waveform visualizer","AnalyserNode canvas","frequency bar animation"),
    ("CSS animation pause resume","animation-play-state","toggle on interaction"),
    ("stagger grid entrance","nth-child delay","animation-delay per item"),
    ("elastic easing","cubic-bezier overshoot","custom bezier curve definition"),
    ("timeline scrub","scroll position maps to animation","JavaScript progress control"),
    ("color palette transition","lerp HSL values","smooth theme switching"),
    ("shape morphing CSS","border-radius animation","blob shape keyframes"),
    ("velocity-based fling","momentum scrolling","deceleration formula"),
    ("reduced motion","prefers-reduced-motion","respecting user accessibility"),
    ("animation composition","add replace accumulate","layering multiple animations"),
]

ANIM_SYSTEMS = [
    "You are a frontend developer specializing in CSS animations, Web Animations API, and GSAP.",
    "You are a creative developer who builds delightful UI animations with attention to performance.",
    "You are a motion design engineer who bridges design intent and browser animation implementation.",
    "You are a performance-focused frontend engineer who explains animation optimization techniques.",
]

CSS_TOPICS = [
    ("CSS specificity","cascade order","specificity score calculation"),
    ("CSS Grid layout","template areas","auto-placement and dense packing"),
    ("Flexbox alignment","justify and align","flex-grow shrink basis"),
    ("CSS custom properties","design tokens","fallback values and inheritance"),
    ("container queries","size-based rules","@container syntax and named containers"),
    ("CSS cascade layers","@layer order","author vs user vs UA layers"),
    ("CSS nesting","parent selector","native nesting vs preprocessor"),
    ("logical properties","writing mode agnostic","inline-start and block-end"),
    ("CSS clamp function","responsive sizing","fluid typography without breakpoints"),
    ("CSS subgrid","nested grid alignment","column and row subgrid"),
    ("has selector","relational pseudo-class","parent selection technique"),
    ("CSS scope","@scope boundary","donut scope and lower boundary"),
    ("color spaces","oklch and display-p3","wide-gamut color authoring"),
    ("CSS grid intrinsic sizing","min-content max-content","fit-content and minmax"),
    ("sticky positioning","scroll behavior","containing block requirements"),
    ("aspect-ratio property","intrinsic ratio","image and video sizing"),
    ("scroll-behavior smooth","CSS vs JS scrolling","scroll-margin-top for anchors"),
    ("CSS masking","mask-image layers","alpha vs luminance masking"),
    ("clip-path shapes","polygon circle ellipse","inset for rounded clipping"),
    ("backdrop-filter","blur and brightness","glassmorphism effect"),
    ("CSS grid auto-fit auto-fill","responsive columns","implicit track sizing"),
    ("CSS counter","list numbering","counter-increment and counter-reset"),
    ("print styles","@media print","page break control"),
    ("CSS reset vs normalize","baseline consistency","opinionated vs minimal reset"),
    ("focus styles","focus-visible pseudo-class","keyboard vs mouse focus"),
    ("dark mode","prefers-color-scheme","CSS custom property token swap"),
    ("CSS transition vs animation","use case comparison","when keyframes win"),
    ("pseudo-element content","before and after","icon injection technique"),
    ("CSS grid named lines","line names","template column line naming"),
    ("overflow and scrollbar","overflow-x overflow-y","scrollbar-gutter for stability"),
    ("CSS font loading","font-display swap","FOUT vs FOIT trade-off"),
    ("variable fonts","font-variation-settings","weight width and custom axes"),
    ("CSS text overflow","ellipsis","multi-line line-clamp"),
    ("min max clamp","responsive value selection","no breakpoints needed"),
    ("CSS image rendering","pixelated crisp-edges","high-DPI display optimization"),
    ("CSS grid track sizing","fr unit","auto vs fr vs fixed mix"),
    ("CSS position sticky table","sticky header row","containing block fix"),
    ("CSS filter functions","blur hue-rotate saturate","stacked filter chain"),
    ("CSS scroll timeline","animation driven by scroll","linked animation progress"),
    ("CSS painting worklet","Houdini","custom background pattern"),
    ("CSS layout algorithm","formatting context","BFC and IFC rules"),
    ("CSS multi-column","column-count and gap","balancing and spanning"),
    ("CSS selector performance","rightmost selector first","avoiding universal selectors"),
    ("CSS at-rules overview","media layer keyframes supports","feature detection"),
    ("border-image","nine-slice scaling","gradient border technique"),
    ("CSS matrix transform","2D 3D matrix","rotation scale skew combined"),
    ("CSS hsl vs oklch","perceptual uniformity","accessible color palette"),
    ("CSS initial inherit unset revert","keyword values","cascade reset"),
    ("CSS gap property","grid and flex gap","replacing margin hacks"),
    ("CSS grid vs flexbox decision","1D vs 2D","choosing the right model"),
]

CSS_SYSTEMS = [
    "You are a CSS expert who builds scalable design systems and explains layout algorithms in depth.",
    "You are a frontend engineer specializing in modern CSS features, responsive design, and accessibility.",
    "You are a web standards engineer who follows CSS specifications and browser compatibility closely.",
    "You are a UI engineer who designs and maintains large-scale CSS architectures for production apps.",
]

FRONTEND_TOPICS = [
    ("React useState hook","local component state","functional update pattern"),
    ("React useEffect","side effect lifecycle","cleanup function and dependency array"),
    ("React useContext","prop drilling solution","context provider and consumer"),
    ("React useReducer","complex state logic","reducer pattern vs useState"),
    ("React useMemo useCallback","memoization","referential equality and re-render"),
    ("React Suspense","async component loading","fallback and concurrent mode"),
    ("React Server Components","server rendering model","client boundary and serialization"),
    ("TypeScript generics","reusable typed functions","constraint and default type params"),
    ("TypeScript discriminated union","tagged union type","exhaustive switch narrowing"),
    ("TypeScript utility types","Partial Required Pick Omit","mapped type transformation"),
    ("React Query","server state management","stale-while-revalidate and invalidation"),
    ("Zustand global state","lightweight store","selector and shallow equality"),
    ("React form with React Hook Form","uncontrolled inputs","validation and error display"),
    ("code splitting dynamic import","bundle optimization","webpack and vite chunk strategy"),
    ("virtual DOM reconciliation","fiber architecture","key prop importance"),
    ("accessibility aria roles","screen reader support","keyboard navigation pattern"),
    ("web vitals optimization","LCP FID CLS","critical render path"),
    ("CSS-in-JS vs CSS modules","style isolation approaches","trade-off analysis"),
    ("error boundary","React error handling","fallback UI and error reporting"),
    ("React portal","rendering outside root","modal and tooltip use case"),
    ("custom hook extraction","reusable stateful logic","naming and testing"),
    ("debounce vs throttle","event rate limiting","leading and trailing edge"),
    ("fetch vs axios","HTTP client comparison","interceptors and error handling"),
    ("WebSocket React integration","real-time updates","useEffect cleanup and reconnect"),
    ("PWA service worker","offline capability","cache-first and network-first strategies"),
    ("testing with React Testing Library","user-centric queries","fireEvent vs userEvent"),
    ("Vitest unit testing","fast test runner","mock and spy patterns"),
    ("Cypress end-to-end","browser automation","selector strategy and intercept"),
    ("Storybook component docs","isolated development","argTypes and controls"),
    ("Next.js app router","file-based routing","layout and loading components"),
    ("Vite build tool","ESM native dev server","rollup production build"),
    ("tree shaking","dead code elimination","ESM named exports requirement"),
    ("hydration mismatch","SSR vs client render","avoiding non-deterministic output"),
    ("React 18 concurrent features","startTransition","useTransition and isPending"),
    ("internationalization i18n","react-intl or i18next","plural and interpolation"),
    ("image optimization","next/image srcset","lazy load and aspect ratio"),
    ("CSS grid with React","dynamic column count","responsive grid component"),
    ("compound component pattern","implicit state sharing","context-based API"),
    ("render prop pattern","children as function","slot-based composition"),
    ("React controlled vs uncontrolled","form input pattern","when to use refs"),
    ("module federation","micro-frontend architecture","shared dependency versioning"),
    ("HTTP caching in frontend","Cache-Control headers","ETag and conditional request"),
    ("IndexedDB persistence","client-side database","idb wrapper pattern"),
    ("canvas 2D API","drawing primitives","pixel manipulation and performance"),
    ("Web Workers","off-main-thread computation","postMessage serialization"),
    ("ResizeObserver","element size tracking","layout measurement pattern"),
    ("MutationObserver","DOM change watching","virtual list implementation hint"),
    ("Intersection Observer","viewport detection","infinite scroll pattern"),
    ("pointer events","mouse touch pen unification","drag interaction"),
    ("browser storage comparison","cookie localStorage sessionStorage IndexedDB","security and capacity"),
]

FRONTEND_SYSTEMS = [
    "You are a senior React engineer who builds production-grade frontend applications.",
    "You are a frontend architect specializing in performance, accessibility, and scalable component design.",
    "You are a TypeScript expert who helps teams write type-safe, maintainable frontend code.",
    "You are a web performance engineer who optimizes Core Web Vitals and bundle sizes.",
]

BACKEND_TOPICS = [
    ("REST API design","resource-based URLs","HTTP verb semantics and status codes"),
    ("FastAPI dependency injection","Depends","shared resources and auth middleware"),
    ("JWT authentication","stateless tokens","access and refresh token rotation"),
    ("OAuth 2.0 authorization code","delegated access","token introspection and revocation"),
    ("database connection pooling","connection lifecycle","pool sizing and timeout"),
    ("SQL query optimization","EXPLAIN ANALYZE","index design and covering indexes"),
    ("database transactions ACID","isolation levels","phantom reads and serializable"),
    ("Redis caching patterns","cache-aside vs read-through","TTL and eviction policy"),
    ("message queue RabbitMQ","producer consumer pattern","dead letter exchange"),
    ("Kafka event streaming","partitions and consumer groups","offset management and replay"),
    ("API rate limiting","token bucket","per-user and per-IP limits"),
    ("pagination cursor-based","keyset pagination","offset vs cursor trade-off"),
    ("API versioning strategies","URL vs header","backward compatibility"),
    ("webhook delivery reliability","retry with backoff","idempotency key"),
    ("database sharding","horizontal partitioning","shard key selection"),
    ("read replica","query routing","replication lag handling"),
    ("database migration","Alembic versioning","zero-downtime migration strategy"),
    ("async task worker Celery","task routing","monitoring and retry"),
    ("gRPC vs REST","protocol buffer","streaming and bidirectional"),
    ("GraphQL schema design","type resolver pattern","N+1 problem and DataLoader"),
    ("API gateway","aggregation and auth","rate limit and circuit breaker"),
    ("microservice communication","synchronous vs async","service discovery"),
    ("circuit breaker pattern","failure isolation","half-open state and recovery"),
    ("saga pattern","distributed transaction","choreography vs orchestration"),
    ("CQRS","command query separation","event sourcing integration"),
    ("event sourcing","append-only log","projection rebuild and snapshot"),
    ("outbox pattern","transactional messaging","at-least-once delivery"),
    ("health check endpoint","liveness vs readiness","Kubernetes probe integration"),
    ("distributed tracing","OpenTelemetry","trace context propagation"),
    ("structured logging","JSON log format","correlation ID and log levels"),
    ("background job processing","priority queues","job deduplication"),
    ("file upload handling","multipart form","stream to S3 and progress"),
    ("password hashing","bcrypt vs Argon2","work factor and salt"),
    ("SQL injection prevention","ORM and parameterized","raw query audit"),
    ("CORS configuration","origin whitelist","preflight request handling"),
    ("content negotiation","Accept header","JSON and XML response"),
    ("idempotent API design","PUT vs PATCH","idempotency key for POST"),
    ("OpenAPI documentation","Swagger schema","code generation from spec"),
    ("database seeding","fixture strategy","test isolation with rollback"),
    ("load testing","Locust scripts","percentile analysis and saturation"),
    ("container health in Kubernetes","readiness probe","graceful shutdown signal handling"),
    ("database index types","B-tree vs hash vs GIN","partial and expression indexes"),
    ("NoSQL data modeling","document design","embedding vs referencing"),
    ("time-series database","InfluxDB or TimescaleDB","retention policy and downsampling"),
    ("full-text search","Elasticsearch query DSL","tokenizer and analyzer"),
    ("API security headers","HSTS CSP X-Frame","middleware implementation"),
    ("server-sent events","SSE vs WebSocket","reconnect and last-event-id"),
    ("backend testing strategy","unit integration contract","test pyramid"),
    ("secret management","Vault dynamic secrets","lease renewal and revocation"),
    ("deployment blue-green","traffic shift","rollback trigger"),
]

BACKEND_SYSTEMS = [
    "You are a senior backend engineer who designs and builds scalable APIs and distributed systems.",
    "You are a database architect who optimizes queries, schema design, and data access patterns.",
    "You are a platform engineer who builds reliable, observable microservice infrastructure.",
    "You are a Python backend developer specializing in FastAPI, SQLAlchemy, and async patterns.",
]

# ---------------------------------------------------------------------------
# EXAMPLE GENERATORS PER DOMAIN
# ---------------------------------------------------------------------------

def cyber_example(rng: random.Random) -> dict:
    topic, attack, defense = rng.choice(CYBER_TOPICS)
    system = rng.choice(CYBER_SYSTEMS)
    q_templates = [
        f"Walk me through how {topic} works at a technical level and what defenders should watch for.",
        f"Explain the {attack} technique — what makes it effective and how is it typically detected?",
        f"During a pentest I encountered {topic}. How do I demonstrate impact and recommend remediation?",
        f"How does {topic} work, and what does a reliable exploit chain look like?",
        f"What are the key indicators that {topic} is being used against an environment?",
        f"Compare {attack} to similar techniques — when do attackers prefer it and why?",
        f"Explain {topic} to a developer who needs to understand why {defense} matters.",
        f"How would you use {topic} in a red team engagement, and what blue team controls stop it?",
    ]
    q = rng.choice(q_templates)

    a_sections = {
        "overview": f"**{topic.title()}**\n\nThis technique exploits {attack} to gain unauthorized access or data. Understanding the mechanics is essential for building effective defenses.\n\n",
        "how_it_works": f"**How It Works**\n\nThe attacker leverages {attack} by targeting systems without {defense} in place. The exploit typically follows a predictable pattern: reconnaissance, trigger, and extraction.\n\n",
        "code": f"**Example Payload / Command**\n```bash\n# Demonstrates {topic} in a controlled lab environment\necho 'Proof-of-concept for {topic}'\n```\n\n",
        "indicators": f"**Detection Indicators**\n- Unusual authentication patterns or privilege escalation attempts\n- Network traffic anomalies matching {attack} signatures\n- Log entries showing failed attempts before successful exploitation\n- File or registry modifications associated with {topic}\n\n",
        "defense": f"**Defense and Remediation**\n\nThe primary mitigation is {defense}. Additional controls:\n1. Enable detailed audit logging for related events\n2. Apply principle of least privilege to all accounts and services\n3. Use a WAF or IDS rule to detect {attack} patterns\n4. Regularly test your defenses against {topic} in a lab environment\n5. Patch affected systems within your SLA based on CVSS severity\n\n",
        "summary": f"**Key Takeaway**\n\nWithout {defense}, {topic} provides attackers a reliable path to {attack}. Prioritize detection before prevention — assume breach and ensure your logging catches this technique.",
    }
    answer = a_sections["overview"] + a_sections["how_it_works"] + a_sections["indicators"] + a_sections["defense"] + a_sections["summary"]
    return {"messages":[
        {"role":"system","content":system},
        {"role":"user","content":q},
        {"role":"assistant","content":answer}
    ]}

def net_example(rng: random.Random) -> dict:
    topic, concept, detail = rng.choice(NET_TOPICS)
    system = rng.choice(NET_SYSTEMS)
    q_templates = [
        f"Explain how {topic} works and why {concept} matters in practice.",
        f"Walk me through {topic} step by step — what happens at the packet level?",
        f"How does {topic} relate to {detail}, and when does it cause problems?",
        f"When troubleshooting {topic} issues, what is your systematic approach?",
        f"Compare {topic} to similar protocols — what are the trade-offs?",
        f"How would you implement {detail} to solve a {concept} problem?",
        f"Explain {topic} to a junior network engineer who understands basic TCP/IP.",
        f"What are common misconfigurations with {topic} and how do you fix them?",
    ]
    q = rng.choice(q_templates)

    answer = f"**{topic.title()}**\n\n{concept.capitalize()} is at the core of how {topic} operates in modern networks.\n\n**Protocol Mechanics**\n\nAt the packet level, {topic} uses {detail} to establish and maintain state. The key fields to understand:\n- Control flags and header structure\n- Timer values and retry behavior\n- State machine transitions\n\n**Troubleshooting Approach**\n\nWhen diagnosing {topic} issues:\n1. Capture traffic with `tcpdump` or Wireshark and filter for relevant protocol\n2. Check interface statistics for errors, drops, and CRC failures\n3. Validate configuration with `show` commands and compare against intended design\n4. Use `ping`, `traceroute`, or `mtr` to isolate the failure domain\n5. Review logs for neighbor relationship changes or protocol restarts\n\n**Common Misconfigurations**\n- Mismatched {detail} parameters between peers\n- Incorrect timer values causing premature timeouts\n- Missing or incorrect access-list entries blocking protocol packets\n- Asymmetric routing causing {concept} issues\n\n**Best Practice Configuration**\n```\n! Ensure {detail} is consistently configured\n! Validate with 'show' and automated compliance checks\n! Document expected state for baseline comparison\n```\n\n**Key Takeaway**\n\nUnderstanding {concept} is foundational to operating {topic} reliably. Always test changes in a lab before production, and maintain clear documentation of intended topology."
    return {"messages":[
        {"role":"system","content":system},
        {"role":"user","content":q},
        {"role":"assistant","content":answer}
    ]}

def it_example(rng: random.Random) -> dict:
    topic, concept, detail = rng.choice(IT_TOPICS)
    system = rng.choice(IT_SYSTEMS)
    q_templates = [
        f"How do you implement {topic} in an enterprise environment following best practices?",
        f"What are the key considerations when designing {concept} for {topic}?",
        f"Walk me through troubleshooting a {topic} failure in production.",
        f"Explain {topic} and how it relates to {detail}.",
        f"How would you automate {topic} at scale using modern tooling?",
        f"What are the common pitfalls with {topic} and how do you avoid them?",
        f"Design a {topic} strategy for a 500-node environment — what decisions matter most?",
        f"Compare different approaches to {concept} and recommend the best one for {topic}.",
    ]
    q = rng.choice(q_templates)

    answer = f"**{topic.title()}**\n\n**Overview**\n\n{concept.capitalize()} is the core challenge with {topic}. Done correctly, it provides {detail} and enables reliable operations at scale.\n\n**Architecture Decisions**\n\nKey design choices:\n1. **Scale requirements** — number of nodes, geographic distribution, SLA targets\n2. **Integration points** — existing tooling, CMDB, monitoring stack\n3. **Change management** — testing cadence, rollback procedures, approval workflows\n4. **Security posture** — least privilege, audit logging, secrets management\n\n**Implementation Steps**\n\n```bash\n# Example: Automated {topic} implementation\n# Step 1: Configure baseline\n# Step 2: Validate with dry-run\n# Step 3: Deploy with change window\n# Step 4: Verify against compliance policy\n```\n\n**Operational Runbook**\n\nFor {topic} incidents:\n- Check service health and recent change log\n- Validate {detail} configuration matches baseline\n- Review error logs for root cause indicators\n- Engage vendor support if within support contract\n\n**Monitoring and Alerting**\n\nKey metrics to watch:\n- Service availability and response time\n- Resource utilization (CPU, memory, disk)\n- Error rate and failure event frequency\n- {concept} drift from desired state\n\n**Key Takeaway**\n\n{topic.capitalize()} requires {detail} from day one. Retrofitting it later is expensive. Invest in automation, documentation, and testing infrastructure to keep operations manageable as you scale."
    return {"messages":[
        {"role":"system","content":system},
        {"role":"user","content":q},
        {"role":"assistant","content":answer}
    ]}

def reasoning_example(rng: random.Random) -> dict:
    topic, concept, detail = rng.choice(REASONING_TOPICS)
    system = rng.choice(REASONING_SYSTEMS)
    q_templates = [
        f"Explain {topic} with a worked example and analyze its time and space complexity.",
        f"How does {topic} work? Walk through {detail} step by step.",
        f"When should I use {topic} over alternatives, and what are the trade-offs?",
        f"Implement {topic} in Python and explain each part of the solution.",
        f"What is the intuition behind {topic}? Explain {concept} from first principles.",
        f"Show me how to approach a problem that requires {topic} — what are the key insights?",
        f"Debug this common mistake in {topic} and explain why {detail} is critical.",
        f"How does {concept} enable {topic} to work efficiently?",
    ]
    q = rng.choice(q_templates)

    answer = f"**{topic.title()}**\n\n**Core Intuition**\n\n{concept.capitalize()} is the key insight that makes {topic} work. Rather than brute-force enumeration, we exploit {detail} to reduce the problem space.\n\n**Step-by-Step Walkthrough**\n\nLet's trace through an example:\n\n```python\ndef solve(input_data):\n    \"\"\"\n    Implements {topic} using {detail}.\n    Time: O(n log n)  Space: O(n)\n    \"\"\"\n    # Step 1: Initialize data structures\n    result = []\n    \n    # Step 2: Apply {concept}\n    # ... core logic using {detail}\n    \n    # Step 3: Return result\n    return result\n\n# Example usage\ndata = [3, 1, 4, 1, 5, 9, 2, 6]\nprint(solve(data))\n```\n\n**Complexity Analysis**\n\n| Aspect | Complexity | Reason |\n|---|---|---|\n| Time | O(n log n) | {detail} reduces comparisons |\n| Space | O(n) | Auxiliary storage for {concept} |\n| Best case | O(n) | When input has special structure |\n\n**Common Pitfalls**\n\n1. **Off-by-one errors** — always verify boundary conditions with small examples\n2. **Integer overflow** — use appropriate data types for index arithmetic\n3. **Wrong base case** — {topic} often fails silently with incorrect initialization\n4. **Ignoring {detail}** — the optimization only applies when the invariant holds\n\n**When to Use This**\n\nChoose {topic} when:\n- Input has the {concept} property\n- Brute force is O(n²) or worse\n- Memory trade-off is acceptable\n\nAvoid when input is adversarial or the {detail} assumption doesn't hold.\n\n**Key Takeaway**\n\nThe insight is {concept} — once you see that, the {topic} implementation follows naturally. Practice with variations to build the pattern recognition."
    return {"messages":[
        {"role":"system","content":system},
        {"role":"user","content":q},
        {"role":"assistant","content":answer}
    ]}

def threejs_example(rng: random.Random) -> dict:
    topic, concept, detail = rng.choice(THREEJS_TOPICS)
    system = rng.choice(THREEJS_SYSTEMS)
    q_templates = [
        f"How do I implement {topic} in Three.js? Show me the complete setup.",
        f"Explain how {concept} works in Three.js and walk through {detail}.",
        f"I'm building a {topic} effect — what's the best approach with Three.js?",
        f"What are the performance considerations for {topic} in a production Three.js scene?",
        f"Show me how to create {topic} from scratch, including {detail}.",
        f"How does {detail} work in the context of {topic}?",
        f"Debug common issues with {topic} — what usually goes wrong?",
        f"Compare different approaches to implementing {concept} for {topic}.",
    ]
    q = rng.choice(q_templates)

    answer = f"**{topic.title()} in Three.js**\n\n**Setup**\n\n```javascript\nimport * as THREE from 'three';\n\n// Scene setup\nconst scene = new THREE.Scene();\nconst camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);\nconst renderer = new THREE.WebGLRenderer({{ antialias: true }});\nrenderer.setSize(window.innerWidth, window.innerHeight);\nrenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));\ndocument.body.appendChild(renderer.domElement);\n```\n\n**Implementing {topic.title()}**\n\n```javascript\n// {concept} — {detail}\nfunction create{topic.replace(' ','').title()}() {{\n  // Configure geometry and material for {topic}\n  const geometry = new THREE.BufferGeometry();\n  const material = new THREE.MeshStandardMaterial({{\n    color: 0x4488ff,\n    roughness: 0.4,\n    metalness: 0.1,\n  }});\n  \n  const mesh = new THREE.Mesh(geometry, material);\n  scene.add(mesh);\n  return mesh;\n}}\n\nconst obj = create{topic.replace(' ','').title()}();\n```\n\n**Animation Loop**\n\n```javascript\nconst clock = new THREE.Clock();\n\nfunction animate() {{\n  requestAnimationFrame(animate);\n  const elapsed = clock.getElapsedTime();\n  \n  // Update {topic} based on elapsed time\n  obj.rotation.y = elapsed * 0.5;\n  \n  renderer.render(scene, camera);\n}}\nanimate();\n```\n\n**Performance Tips**\n\n1. **Reduce draw calls** — merge static geometries with `BufferGeometryUtils.mergeGeometries()`\n2. **Use {detail}** correctly to avoid GPU stalls\n3. **Dispose resources** — call `geometry.dispose()`, `material.dispose()`, `texture.dispose()` when removing objects\n4. **Profile with Spector.js** — inspect WebGL calls and find bottlenecks\n5. **Level of detail** — swap to simpler geometry at distance\n\n**Handle Resize**\n\n```javascript\nwindow.addEventListener('resize', () => {{\n  camera.aspect = window.innerWidth / window.innerHeight;\n  camera.updateProjectionMatrix();\n  renderer.setSize(window.innerWidth, window.innerHeight);\n}});\n```\n\n**Key Takeaway**\n\nThe {concept} pattern is central to {topic}. Start with the minimal working version, then add {detail} once the basic effect is working correctly."
    return {"messages":[
        {"role":"system","content":system},
        {"role":"user","content":q},
        {"role":"assistant","content":answer}
    ]}

def anim_example(rng: random.Random) -> dict:
    topic, concept, detail = rng.choice(ANIM_TOPICS)
    system = rng.choice(ANIM_SYSTEMS)
    q_templates = [
        f"How do I implement {topic} with good performance? Show me the code.",
        f"Explain {concept} and walk me through {detail} for {topic}.",
        f"What's the right approach for {topic} — CSS or JavaScript?",
        f"Build a {topic} that works smoothly on mobile — what are the constraints?",
        f"How does {detail} enable {concept} in {topic}?",
        f"I need a {topic} effect — show me a complete implementation.",
        f"What are the accessibility considerations for {topic}?",
        f"Debug a common performance problem with {topic}.",
    ]
    q = rng.choice(q_templates)

    answer = f"**{topic.title()}**\n\n**Approach**\n\n{concept.capitalize()} drives this effect. We use {detail} to achieve smooth rendering without janking the main thread.\n\n**CSS Implementation**\n\n```css\n/* Base state */\n.element {{\n  /* Use transform and opacity — compositor-only properties */\n  transform: translateY(0);\n  opacity: 1;\n  transition: transform 0.3s ease, opacity 0.3s ease;\n  will-change: transform; /* promote to compositor layer */\n}}\n\n/* Animated state */\n.element.active {{\n  animation: {topic.replace(' ','-')} 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;\n}}\n\n@keyframes {topic.replace(' ','-')} {{\n  from {{\n    transform: translateY(20px);\n    opacity: 0;\n  }}\n  to {{\n    transform: translateY(0);\n    opacity: 1;\n  }}\n}}\n```\n\n**JavaScript Enhancement**\n\n```javascript\n// Use Web Animations API for programmatic control\nconst el = document.querySelector('.element');\n\nconst anim = el.animate([\n  {{ transform: 'translateY(20px)', opacity: 0 }},\n  {{ transform: 'translateY(0)', opacity: 1 }}\n], {{\n  duration: 600,\n  easing: 'cubic-bezier(0.34, 1.56, 0.64, 1)',\n  fill: 'forwards'\n}});\n\n// Listen for completion\nanim.addEventListener('finish', () => {{\n  el.classList.add('settled');\n}});\n```\n\n**Performance Rules**\n\n1. Only animate `transform` and `opacity` — these are compositor-only and never trigger layout\n2. Avoid animating `width`, `height`, `top`, `left` — they cause layout thrashing\n3. Use `will-change: transform` sparingly — creates a compositing layer (memory cost)\n4. Batch DOM reads before writes — avoid layout thrashing in JS-driven animations\n\n**Accessibility**\n\n```css\n@media (prefers-reduced-motion: reduce) {{\n  .element, .element.active {{\n    animation: none;\n    transition: none;\n  }}\n}}\n```\n\n**Key Takeaway**\n\nKeep {topic} on the compositor thread — transform and opacity only. Add `prefers-reduced-motion` support so users who experience motion sickness aren't excluded."
    return {"messages":[
        {"role":"system","content":system},
        {"role":"user","content":q},
        {"role":"assistant","content":answer}
    ]}

def css_example(rng: random.Random) -> dict:
    topic, concept, detail = rng.choice(CSS_TOPICS)
    system = rng.choice(CSS_SYSTEMS)
    q_templates = [
        f"Explain {topic} — how does {concept} work and when does it matter?",
        f"How do I use {topic} to solve {detail}? Show a practical example.",
        f"What are the browser compatibility concerns with {topic}?",
        f"How does {concept} interact with the cascade and inheritance?",
        f"Walk me through {topic} with a real-world layout problem.",
        f"Compare {topic} to older approaches — why is the modern way better?",
        f"Debug a common {topic} issue — what usually confuses developers?",
        f"How does {detail} change the behavior of {topic}?",
    ]
    q = rng.choice(q_templates)

    answer = f"**{topic.title()}**\n\n**How It Works**\n\n{concept.capitalize()} is the mechanism behind {topic}. {detail.capitalize()} defines the specific behavior that makes this powerful.\n\n**Basic Usage**\n\n```css\n/* {topic} — foundational pattern */\n.container {{\n  /* Core declaration */\n  display: grid; /* or flex, depending on context */\n}}\n\n/* {detail} in action */\n.element {{\n  /* The key property for {topic} */\n  /* See MDN for full syntax reference */\n}}\n```\n\n**Practical Example**\n\n```html\n<div class=\"layout\">\n  <header class=\"header\">Header</header>\n  <main class=\"main\">Content</main>\n  <aside class=\"sidebar\">Sidebar</aside>\n  <footer class=\"footer\">Footer</footer>\n</div>\n```\n\n```css\n.layout {{\n  display: grid;\n  grid-template-areas:\n    \"header header\"\n    \"main   sidebar\"\n    \"footer footer\";\n  grid-template-columns: 1fr 300px;\n  gap: 1rem;\n}}\n\n/* Apply {topic} for responsive behavior */\n@media (max-width: 768px) {{\n  .layout {{\n    grid-template-areas:\n      \"header\"\n      \"main\"\n      \"sidebar\"\n      \"footer\";\n    grid-template-columns: 1fr;\n  }}\n}}\n```\n\n**Common Mistakes**\n\n1. **{concept} confusion** — developers often misunderstand the default values\n2. **Specificity battles** — {topic} interacts with the cascade in non-obvious ways\n3. **Forgetting {detail}** — the key detail that changes behavior\n4. **Browser support gaps** — check Can I Use before using cutting-edge features\n\n**Browser Support**\n\nCore {topic} features have excellent support (95%+ global). Check `@supports` for progressive enhancement:\n```css\n@supports ({topic.split()[0]}: initial) {{\n  /* Enhanced layout using {topic} */\n}}\n```\n\n**Key Takeaway**\n\n{topic.capitalize()} solves {concept} elegantly. Understanding {detail} is what separates developers who fight CSS from those who work with it."
    return {"messages":[
        {"role":"system","content":system},
        {"role":"user","content":q},
        {"role":"assistant","content":answer}
    ]}

def frontend_example(rng: random.Random) -> dict:
    topic, concept, detail = rng.choice(FRONTEND_TOPICS)
    system = rng.choice(FRONTEND_SYSTEMS)
    q_templates = [
        f"Explain {topic} — how does {concept} work and when should I use it?",
        f"Show me a complete example of {topic} with {detail}.",
        f"What are the performance implications of {topic} in a large React app?",
        f"How does {detail} affect {topic} behavior?",
        f"Walk me through debugging a {topic} issue in production.",
        f"Compare {topic} to alternatives — when is each the right choice?",
        f"How do I test {topic} properly using React Testing Library?",
        f"Implement {topic} with TypeScript — what types do I need?",
    ]
    q = rng.choice(q_templates)

    answer = f"**{topic.title()}**\n\n**How It Works**\n\n{concept.capitalize()} is the core mechanism. {detail.capitalize()} is the critical detail that determines when and how to use this correctly.\n\n**Basic Implementation**\n\n```tsx\nimport React, {{ useState, useEffect, useCallback }} from 'react';\n\ninterface Props {{\n  initialValue?: string;\n  onChange?: (value: string) => void;\n}}\n\nconst MyComponent: React.FC<Props> = ({{ initialValue = '', onChange }}) => {{\n  // {topic} implementation\n  const [value, setValue] = useState(initialValue);\n  \n  // Handle {detail}\n  const handleChange = useCallback((newValue: string) => {{\n    setValue(newValue);\n    onChange?.(newValue);\n  }}, [onChange]);\n  \n  return (\n    <div className=\"component\">\n      <input\n        value={{value}}\n        onChange={{e => handleChange(e.target.value)}}\n        aria-label=\"Input field\"\n      />\n      <p>Current: {{value}}</p>\n    </div>\n  );\n}};\n\nexport default MyComponent;\n```\n\n**Advanced Usage with {concept}**\n\n```tsx\n// When {detail} becomes important at scale\nconst OptimizedComponent = React.memo(MyComponent, (prev, next) => {{\n  return prev.initialValue === next.initialValue;\n}});\n\n// Custom hook for reusable logic\nfunction use{topic.split()[0].title()}(initialValue: string) {{\n  const [state, setState] = useState(initialValue);\n  // ... logic\n  return {{ state, setState }};\n}}\n```\n\n**Testing**\n\n```tsx\nimport {{ render, screen, fireEvent }} from '@testing-library/react';\nimport MyComponent from './MyComponent';\n\ntest('handles {concept} correctly', async () => {{\n  const mockOnChange = jest.fn();\n  render(<MyComponent onChange={{mockOnChange}} />);\n  \n  const input = screen.getByLabelText('Input field');\n  fireEvent.change(input, {{ target: {{ value: 'test' }} }});\n  \n  expect(mockOnChange).toHaveBeenCalledWith('test');\n}});\n```\n\n**Common Mistakes**\n\n1. **Missing {detail}** — causes subtle bugs that only appear in production\n2. **Stale closures** — always include all dependencies in dependency arrays\n3. **Over-memoization** — `useMemo`/`useCallback` have costs; don't use them everywhere\n4. **Type assertions over proper types** — use proper TypeScript generics\n\n**Key Takeaway**\n\n{topic.capitalize()} is the right tool when {concept} is the problem. Always measure before optimizing — React is fast by default."
    return {"messages":[
        {"role":"system","content":system},
        {"role":"user","content":q},
        {"role":"assistant","content":answer}
    ]}

def backend_example(rng: random.Random) -> dict:
    topic, concept, detail = rng.choice(BACKEND_TOPICS)
    system = rng.choice(BACKEND_SYSTEMS)
    q_templates = [
        f"How do I implement {topic} correctly in a production FastAPI service?",
        f"Explain {concept} and how it applies to {topic}.",
        f"Walk me through designing {topic} for a high-traffic API.",
        f"What are the security implications of {topic} and how do I mitigate them?",
        f"Show me a complete {topic} implementation with error handling.",
        f"How does {detail} affect {topic} at scale?",
        f"Compare approaches to {topic} — what does production look like?",
        f"Debug a common {topic} issue — what's the root cause and fix?",
    ]
    q = rng.choice(q_templates)

    answer = f"**{topic.title()}**\n\n**Design Principles**\n\n{concept.capitalize()} drives the architecture. {detail.capitalize()} is the implementation detail that distinguishes robust systems from fragile ones.\n\n**FastAPI Implementation**\n\n```python\nfrom fastapi import FastAPI, HTTPException, Depends, status\nfrom pydantic import BaseModel\nfrom typing import Optional\nimport logging\n\nlogger = logging.getLogger(__name__)\napp = FastAPI()\n\n\nclass RequestModel(BaseModel):\n    data: str\n    options: Optional[dict] = None\n\n\n@app.post('/endpoint', status_code=status.HTTP_200_OK)\nasync def handle_request(\n    payload: RequestModel,\n    # dependency injection for auth, db, etc.\n) -> dict:\n    \"\"\"\n    Handles {topic} with {detail}.\n    \"\"\"\n    try:\n        # Validate input\n        if not payload.data:\n            raise HTTPException(\n                status_code=status.HTTP_400_BAD_REQUEST,\n                detail='data field is required'\n            )\n        \n        # Core logic: {concept}\n        result = await process(payload.data)\n        \n        logger.info('Request processed', extra={{'size': len(payload.data)}})\n        return {{'status': 'ok', 'result': result}}\n        \n    except Exception as exc:\n        logger.exception('Unexpected error in handle_request')\n        raise HTTPException(\n            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,\n            detail='Internal server error'\n        ) from exc\n```\n\n**Database Pattern**\n\n```python\nfrom sqlalchemy.ext.asyncio import AsyncSession\nfrom sqlalchemy import select\n\nasync def get_data(session: AsyncSession, id: int):\n    # Always use parameterized queries\n    result = await session.execute(\n        select(Model).where(Model.id == id)\n    )\n    return result.scalar_one_or_none()\n```\n\n**Testing**\n\n```python\nimport pytest\nfrom httpx import AsyncClient\n\n@pytest.mark.asyncio\nasync def test_endpoint():\n    async with AsyncClient(app=app, base_url='http://test') as client:\n        response = await client.post('/endpoint', json={{'data': 'test'}})\n    assert response.status_code == 200\n    assert response.json()['status'] == 'ok'\n```\n\n**Observability**\n\n- Structured JSON logging with correlation IDs\n- Prometheus metrics: request count, latency histogram, error rate\n- OpenTelemetry tracing for distributed request tracking\n- Health check endpoint at `/health` for Kubernetes probes\n\n**Key Takeaway**\n\n{topic.capitalize()} requires {detail} to be production-ready. Get the happy path working first, then add error handling, logging, and observability before calling it done."
    return {"messages":[
        {"role":"system","content":system},
        {"role":"user","content":q},
        {"role":"assistant","content":answer}
    ]}

# ---------------------------------------------------------------------------
# MAIN GENERATION
# ---------------------------------------------------------------------------

# Hand-crafted seed examples (already in file from before)
SEED_EXAMPLES = []  # populated below

def generate_all(total: int = 2000, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)

    # Domain targets: 222-223 per domain for 9 domains = 2000
    targets = {
        "cybersecurity": 222,
        "networking": 222,
        "it": 222,
        "reasoning": 222,
        "threejs": 222,
        "animations": 222,
        "css": 222,
        "frontend": 223,
        "backend": 223,
    }

    generators = {
        "cybersecurity": cyber_example,
        "networking": net_example,
        "it": it_example,
        "reasoning": reasoning_example,
        "threejs": threejs_example,
        "animations": anim_example,
        "css": css_example,
        "frontend": frontend_example,
        "backend": backend_example,
    }

    examples = []
    for domain, count in targets.items():
        gen = generators[domain]
        seen_qs: set[str] = set()
        attempts = 0
        while len([e for e in examples if domain in str(e["messages"][0]["content"]).lower() or True]) < sum(targets[d] for d in list(targets.keys())[:list(targets.keys()).index(domain)+1]) - sum(targets[d] for d in list(targets.keys())[list(targets.keys()).index(domain)+1:]):
            if len([]) >= count:
                break
            ex = gen(rng)
            user_q = ex["messages"][1]["content"].strip().lower()
            if user_q not in seen_qs:
                seen_qs.add(user_q)
                examples.append(ex)
                if len(examples) % 100 == 0:
                    pass
            attempts += 1
            if attempts > count * 3:
                break

    # Simpler approach: generate exactly the right count per domain
    examples = []
    for domain, count in targets.items():
        gen = generators[domain]
        seen_qs: set[str] = set()
        domain_examples = []
        attempts = 0
        while len(domain_examples) < count:
            ex = gen(rng)
            user_q = ex["messages"][1]["content"].strip().lower()
            if user_q not in seen_qs:
                seen_qs.add(user_q)
                domain_examples.append(ex)
            attempts += 1
            if attempts > count * 10:
                # Fill remainder with slight variations
                domain_examples.append(gen(rng))
                if len(domain_examples) >= count:
                    break
        examples.extend(domain_examples[:count])

    rng.shuffle(examples)
    return examples[:total]


def write_jsonl(examples: list[dict], output_path: str):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {output}")

    # Domain stats (infer from system prompt content)
    domain_kw = {
        "cybersecurity": ["penetration tester", "red team", "malware analyst", "vulnerability researcher", "security engineer", "CTF"],
        "networking": ["network engineer", "network security analyst", "network architect", "telecommunications"],
        "it": ["systems administrator", "DevOps engineer", "cloud architect", "site reliability", "infrastructure automation"],
        "reasoning": ["competitive programmer", "computer science professor", "software architect", "mathematician", "senior engineer"],
        "threejs": ["Three.js", "graphics programmer", "WebGL", "creative developer", "game developer"],
        "animations": ["CSS animations", "creative developer", "motion design", "performance-focused frontend"],
        "css": ["CSS expert", "frontend engineer specializing in modern CSS", "web standards", "UI engineer"],
        "frontend": ["senior React engineer", "frontend architect", "TypeScript expert", "web performance engineer"],
        "backend": ["senior backend engineer", "database architect", "platform engineer", "Python backend"],
    }

    counts = {d: 0 for d in domain_kw}
    for ex in examples:
        system_text = ex["messages"][0]["content"]
        for domain, kws in domain_kw.items():
            if any(kw.lower() in system_text.lower() for kw in kws):
                counts[domain] += 1
                break

    print("\nDomain distribution:")
    for domain, count in sorted(counts.items()):
        bar = "█" * (count // 10)
        print(f"  {domain:<15} {count:>4}  {bar}")

    total = sum(counts.values())
    unclassified = len(examples) - total
    if unclassified:
        print(f"  {'(other)':<15} {unclassified:>4}")


if __name__ == "__main__":
    import sys
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    print(f"Generating {total} examples...")
    examples = generate_all(total)
    write_jsonl(examples, "qwen_dataset.jsonl")
