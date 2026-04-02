"""Save Q491-Q495 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (491, "Medicine", "What is the microbiome and how does it affect health?",
     "The microbiome: the ~38 trillion microorganisms (bacteria, fungi, archaea, viruses) living in and on the human body -- roughly equal to the number of human cells. The gut microbiome is the most studied.\n\n"
     "**What it does**:\n"
     "1. **Digestion**: breaks down complex carbohydrates, fibres, produces short-chain fatty acids (SCFAs -- butyrate, propionate, acetate) that feed colonocytes and regulate inflammation.\n"
     "2. **Immune training**: ~70% of immune system is gut-associated. Microbiome educates immune cells to distinguish friend from foe. Germ-free mice have severely underdeveloped immune systems.\n"
     "3. **Neurotransmitters**: gut produces ~90% of body's serotonin. Gut-brain axis -- vagus nerve carries signals bidirectionally. Dysbiosis linked to depression, anxiety.\n"
     "4. **Vitamin synthesis**: B12, K2, some B vitamins.\n"
     "5. **Pathogen resistance**: competitive exclusion -- colonised gut harder for pathogens to invade (C. diff flourishes when antibiotics wipe out competition).\n\n"
     "**Dysbiosis and disease**: disrupted microbiome linked to IBD, IBS, obesity, Type 2 diabetes, colorectal cancer, autism (correlation, causation unclear), depression. Antibiotic overuse = major disruptor.\n\n"
     "**Faecal microbiota transplant (FMT)**: transfer of donor stool to recipient. 90% cure rate for recurrent C. diff infection -- best evidence of microbiome causation. Clinical trials ongoing for IBD, obesity, cancer.\n\n"
     "**What builds a healthy microbiome**: fibre diversity (30 different plants per week -- Tim Spector research); fermented foods (yogurt, kefir, kimchi -- Sonnenburg 2021 RCT shows increased microbiome diversity); avoid unnecessary antibiotics; vaginal birth and breastfeeding inoculate infant microbiome.\n\n"
     "**Caveats**: most microbiome research is correlational. Probiotic supplements have weak evidence for healthy adults. The field is young -- many claims outpace evidence.",
     9.0, "Q491 Microbiome: 38T organisms, gut-brain axis, 90% serotonin, immune education; FMT 90% for C.diff; dysbiosis linked to IBD/obesity/depression; fibre diversity key"),

    (492, "Physics", "What is a black hole and what happens at the singularity?",
     "A black hole: a region of space-time where gravity is so strong that nothing -- not even light -- can escape once past the event horizon.\n\n"
     "**Formation**: massive stars (>20 solar masses) exhaust nuclear fuel, core collapses under gravity. If remaining mass exceeds ~3 solar masses (Tolman-Oppenheimer-Volkoff limit), nothing stops the collapse -- black hole. Also: supermassive black holes (millions to billions of solar masses) at galaxy centres (Sagittarius A* = 4 million solar masses at Milky Way centre).\n\n"
     "**Event horizon**: the point of no return. Not a physical surface -- you'd feel nothing crossing it (locally). But your future light cone points only inward. Radius = Schwarzschild radius: r = 2GM/c^2. For Earth's mass: ~9mm.\n\n"
     "**Hawking radiation** (1974): quantum effects near event horizon cause black holes to emit thermal radiation and slowly evaporate. Temperature inversely proportional to mass -- stellar black holes are colder than CMB and essentially don't evaporate on cosmic timescales. Micro black holes would evaporate instantly.\n\n"
     "**The singularity**: general relativity predicts infinite density, infinite curvature at the centre. This is a breakdown of the theory, not a physical reality. Singularity = where GR stops working. Quantum gravity (string theory, loop quantum gravity) should resolve it -- but we don't have the theory yet.\n\n"
     "**Information paradox**: Hawking radiation is thermal (random). If black hole evaporates completely, information about what fell in is destroyed -- violating quantum mechanics. Still unresolved. Possible resolution: information is encoded in Hawking radiation (subtle correlations), or in a 'remnant.'",
     10.0, "Q492 Black holes: event horizon (Schwarzschild radius), formation from massive stars; Hawking radiation; singularity = GR breakdown not physical; information paradox unresolved"),

    (493, "Sociology", "What causes crime, and does punishment actually reduce it?",
     "Crime is not caused by a single factor -- it emerges from interaction of individual, social, and structural variables.\n\n"
     "**Individual-level factors**: low self-control (Gottfredson & Hirschi), antisocial personality, substance abuse, mental illness (small effect -- most mentally ill are not violent; most violent are not mentally ill).\n\n"
     "**Social/environmental factors**:\n"
     "1. **Social disorganisation theory** (Shaw & McKay 1942): crime concentrates in neighbourhoods with weak social ties, high turnover, poverty -- not in people.\n"
     "2. **Strain theory** (Merton): crime results from gap between culturally valued goals (wealth, success) and legitimate means to achieve them. Innovate (crime) when blocked.\n"
     "3. **Social learning**: crime learned in peer groups (differential association -- Sutherland).\n"
     "4. **Routine activity theory**: crime occurs when motivated offender + suitable target + absent guardian converge.\n\n"
     "**Structural factors**: poverty, inequality (Gini coefficient predicts crime better than poverty alone), unemployment, housing instability, childhood adversity (ACEs), neighbourhood disinvestment.\n\n"
     "**Does punishment reduce crime?**\n"
     "- **Certainty > severity**: research consistently shows certainty of being caught deters more than harshness of punishment. Most offenders don't think they'll be caught.\n"
     "- **Death penalty**: no reliable deterrent effect (states with/without have similar homicide rates).\n"
     "- **Mass incarceration**: US incarcerates 2M people (highest per capita globally). Criminologists broadly agree: past a threshold, more imprisonment increases crime (family disruption, stigma, criminal networks). Returns to prison are close to zero after ~1980s.\n"
     "- **What works**: early childhood intervention (Perry Preschool -- ROI 7-12x), drug treatment, cognitive-behavioural therapy, community policing, environmental design (crime decreases with more streetlights).",
     9.0, "Q493 Crime: social disorganisation, strain theory, routine activity; certainty > severity of punishment; mass incarceration counterproductive; early intervention works best"),

    (494, "Technology", "How does the internet actually work?",
     "The internet: a global network of interconnected networks using standardised protocols to route and deliver data.\n\n"
     "**Physical layer**: fibre optic cables (light pulses -- transatlantic cables carry terabits/second); copper (DSL, coaxial); wireless (WiFi, 4G/5G -- radio waves). Everything is ultimately photons or electrons.\n\n"
     "**IP (Internet Protocol)**: every device has an IP address (IPv4: 32-bit = ~4 billion addresses, nearly exhausted; IPv6: 128-bit = 340 undecillion addresses). Data broken into packets (~1,500 bytes each), each labelled with source and destination IP.\n\n"
     "**Routing**: packets travel through routers -- each router has a routing table and forwards the packet toward the destination using BGP (Border Gateway Protocol). Packets may take different paths and reassemble at destination.\n\n"
     "**TCP (Transmission Control Protocol)**: guarantees delivery and order. Three-way handshake (SYN, SYN-ACK, ACK). Receiver acknowledges packets; sender retransmits lost ones. Slower but reliable. Used for web, email.\n\n"
     "**UDP**: no handshake, no acknowledgement. Fast but lossy. Used for video streaming, gaming, VoIP -- better to drop a frame than freeze.\n\n"
     "**DNS (Domain Name System)**: translates human-readable names (google.com) to IP addresses. Hierarchical: root servers -> TLD servers (.com, .org) -> authoritative servers. Your browser caches results.\n\n"
     "**HTTPS**: HTTP + TLS encryption. TLS handshake: server sends certificate (signed by trusted CA); client verifies; they negotiate encryption keys using asymmetric cryptography (RSA/ECDH); subsequent traffic encrypted with symmetric key (AES). The padlock in your browser.\n\n"
     "**What you actually see**: when you type a URL, browser does DNS lookup, opens TCP connection, sends HTTP request, server sends back HTML/CSS/JS, browser renders. The whole process in ~100 milliseconds.",
     9.0, "Q494 Internet: IP addresses + packets + routing (BGP); TCP (reliable) vs UDP (fast); DNS (name->IP); HTTPS = TLS encryption; browser-to-page in ~100ms"),

    (495, "Mathematics", "What is calculus and why was it such a revolutionary invention?",
     "Calculus: the mathematics of continuous change. Two branches -- differential calculus (rates of change) and integral calculus (accumulation) -- linked by the Fundamental Theorem of Calculus.\n\n"
     "**Differential calculus**: the derivative. If f(x) describes position, f'(x) = velocity, f''(x) = acceleration. Formally: the limit of (f(x+h)-f(x))/h as h->0. The slope of the tangent line at a point.\n\n"
     "**Integral calculus**: the integral. Area under a curve, total accumulation. If you know velocity, integrate to get distance. Antiderivative of f'(x) is f(x) + C.\n\n"
     "**Fundamental Theorem of Calculus**: differentiation and integration are inverse operations. Revolutionary -- linked two seemingly unrelated problems.\n\n"
     "**History**: Newton and Leibniz independently invented calculus (1660s-1680s). Bitter priority dispute (Newton claimed first, published later; Leibniz published first). Modern notation (dy/dx, integral sign) is Leibniz's. Newton's dot notation still used in physics.\n\n"
     "**Why it was revolutionary**:\n"
     "1. **Physics**: Newton's laws of motion use derivatives (F=ma = F=m*d^2x/dt^2). Without calculus, no mechanics, no orbital calculation, no space travel.\n"
     "2. **Engineering**: stress in beams, fluid flow, heat transfer -- all calculus.\n"
     "3. **Electricity and magnetism**: Maxwell's equations (calculus) predicted radio waves before Hertz discovered them experimentally.\n"
     "4. **Economics**: marginal utility, marginal cost, optimisation -- economics runs on calculus.\n"
     "5. **Machine learning**: backpropagation is just calculus (chain rule for partial derivatives).\n\n"
     "**Limits and rigor**: Newton and Leibniz used 'infinitesimals' intuitively. Cauchy and Weierstrass (1800s) put it on rigorous footing with epsilon-delta definition of limits -- removing the 'ghost of departed quantities' that Berkeley mocked.",
     9.5, "Q495 Calculus: derivatives (rates) + integrals (accumulation) linked by FTC; Newton vs Leibniz; revolutionary for physics (F=ma), engineering, Maxwell's equations, ML backprop"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total: {total}")