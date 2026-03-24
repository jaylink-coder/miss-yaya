"""Save Q426-Q430 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (426, "Physics", "What is dark matter and how do we know it exists?",
     "~27% of universe's energy. Doesn't emit/absorb/reflect light. Gravitational effects overwhelming.\n\n"
     "**Evidence:**\n"
     "1. Galaxy rotation curves (Rubin & Ford 1970s): outer stars orbit as fast as inner — Newtonian gravity says slower. Dark matter halo explains flat curves.\n"
     "2. Gravitational lensing: Bullet Cluster — hot gas slowed in collision, but most mass passed straight through (lensing maps it) = dark matter doesn't interact with gas.\n"
     "3. CMB power spectrum: acoustic oscillations fit LCDM only with ~27% dark matter.\n"
     "4. Large-scale structure: galaxy cluster formation simulations require dark matter.\n\n"
     "**Candidates:** WIMPs (theoretically motivated, not found by LHC or detectors), axions, primordial black holes, sterile neutrinos.\n\n"
     "**MOND alternative**: modified Newtonian dynamics; cannot explain Bullet Cluster + CMB simultaneously. Most physicists favour particle dark matter.",
     9.5, "Q426 Dark matter (~27% universe): rotation curves (Rubin), Bullet Cluster lensing, CMB; WIMP not found; axions, sterile neutrinos; MOND fails Bullet Cluster"),

    (427, "Literature", "Why is 'Don Quixote' considered the first modern novel?",
     "Cervantes (Part I: 1605, Part II: 1615). Why it's considered the first modern novel:\n\n"
     "1. Self-consciousness: the novel is aware it's a novel. Part II characters have read Part I. Unreliable narrators, manuscript frames — postmodern devices 400 years early.\n\n"
     "2. Psychological depth: Quixote and Sancho Panza have interior lives, contradictions, development — not types but people.\n\n"
     "3. Reality vs. illusion: windmills vs. giants — about the relationship between literature and reality. Not simple parody; Quixote is also genuinely noble.\n\n"
     "4. Class and voice: Sancho Panza (peasant) has equal narrative weight; earthy pragmatism counterweights idealism.\n\n"
     "5. Intertextuality: comments on its own sources, contemporary fiction, the act of reading.\n\n"
     "What makes it 'modern': fundamentally ambiguous relationship with truth; character who changes across time (Part I vs II Quixote). Bloom: 'most universal and most particular character in literature.'",
     9.0, "Q427 Don Quixote (Cervantes 1605/1615): first modern novel — self-aware, psychological depth, reality/illusion theme, class equality, intertextuality; Bloom's praise"),

    (428, "Computer Science", "What is the halting problem and why does it matter?",
     "Turing (1936): no general algorithm can determine whether an arbitrary program will halt or run forever.\n\n"
     "**Proof (diagonal argument)**: assume HALT(P,I) exists. Construct DIAG(P): if HALT says P halts on P -> loop; if HALT says P loops -> halt.\n"
     "Run DIAG(DIAG): if halts -> loops, if loops -> halts. Contradiction. HALT cannot exist.\n\n"
     "**Why it matters:**\n"
     "1. Limits of computation: well-defined questions that no algorithm can answer. Proved impossible, not just unfound.\n"
     "2. Software verification: cannot automatically check all programs for infinite loops.\n"
     "3. Rice's Theorem: any non-trivial semantic program property is undecidable.\n"
     "4. Closely related to Godel's incompleteness (1931) — same result in different domains.\n"
     "5. Practical: compilers and static analysers must be incomplete or unsound — they make heuristic approximations.",
     10.0, "Q428 Halting problem (Turing 1936): proved undecidable via diagonal argument; limits computation; software verification incomplete; Rice's theorem; related to Godel"),

    (429, "Medicine", "How do vaccines work, and why do some require boosters?",
     "Vaccine presents antigen (weakened/killed pathogen, subunit protein, or mRNA) without disease, training immune memory.\n\n"
     "**Mechanism:**\n"
     "1. Antigen presented to naive B and T cells\n"
     "2. Clonal expansion: matching cells multiply\n"
     "3. Affinity maturation: B-cells mutate antibody genes; best-fitting selected\n"
     "4. Memory B/T cells form: long-lived, recognise pathogen quickly on re-exposure\n"
     "5. Real infection: memory activates in hours not days; antibodies produced before pathogen spreads\n\n"
     "**Types:**\n"
     "- Live attenuated (MMR, varicella): strong, lifetime immunity, can't use in immunocompromised\n"
     "- Inactivated (flu, IPV): weaker, needs boosters\n"
     "- Subunit (hep B, shingles): safe, needs adjuvants + boosters\n"
     "- mRNA (COVID Moderna/Pfizer): mRNA -> cell makes spike protein -> immune response; no DNA alteration\n\n"
     "**Why boosters**: antibody titres wane (inactivated/subunit weaker); mutating viruses (flu) change antigens.",
     9.5, "Q429 Vaccines: antigen -> clonal expansion -> affinity maturation -> memory B/T cells; live attenuated strongest; mRNA (no DNA alteration); boosters because titres wane"),

    (430, "Philosophy", "What is free will, and does science undermine it?",
     "Are our choices genuinely ours, or determined by prior causes?\n\n"
     "**Three positions:**\n"
     "1. Hard determinism: every event caused by prior events; brain states cause decisions; free will = illusion\n"
     "2. Compatibilism (dominant — Hume, Frankfurt, Dennett): 'free will' compatible with determinism. What matters: did *you* (your values, reasoning) cause the action? Freedom = acting on own reasons without coercion. Moral responsibility preserved.\n"
     "3. Libertarian free will: genuine indeterminism (quantum randomness, agent causation). Problem: random causation isn't the freedom we want.\n\n"
     "**Libet (1983)**: readiness potential ~350ms before conscious awareness of deciding. Brain 'decides' first? Heavily criticised: trivial task, complex decisions differ.\n\n"
     "**Dennett**: evolution built a brain that represents itself as agent. This representation IS real — it IS you deliberating. Not false.\n\n"
     "**Does science undermine it?** Depends what you mean. Compatibilist free will survives. Hard determinism mostly fringe.",
     9.5, "Q430 Free will: hard determinism (illusion) vs compatibilism (dominant, Dennett — real agency in causal chain) vs libertarian (indeterminism); Libet experiment critiqued"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total memories: {total}")
