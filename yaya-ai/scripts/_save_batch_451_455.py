"""Save Q451-Q455 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (451, "Geology", "How do earthquakes happen, and what determines their destructiveness?",
     "Earthquakes occur when stress accumulated along fault lines releases suddenly, sending seismic waves through the Earth.\n\n"
     "**Plate tectonics basis**: Earth's lithosphere is divided into ~15 major plates. At boundaries, plates converge, diverge, or slide past each other. Friction locks faults -- stress builds for decades, centuries. When friction is overcome, fault slips -- earthquake.\n\n"
     "**Types of faults**: Strike-slip (San Andreas -- horizontal motion); thrust/reverse (Himalayas -- compression); normal (Basin and Range -- extension). Subduction zones (oceanic plate under continental) produce megathrust quakes -- the largest type (Tohoku 2011, Mw 9.0; Valdivia 1960, Mw 9.5).\n\n"
     "**Magnitude**: Moment magnitude (Mw) measures total energy released. Log scale -- Mw 7 releases ~32x more energy than Mw 6. Mw 9 is approx. 1000x Mw 7.\n\n"
     "**Destructiveness factors**:\n"
     "1. Depth: shallow (<70km) = more damage at surface\n"
     "2. Distance from epicentre\n"
     "3. Soil type: soft sediment amplifies shaking (Mexico City 1985)\n"
     "4. Building quality: 2010 Haiti (Mw 7.0) killed 200k; 2011 Christchurch (Mw 6.3) killed 185 -- same magnitude, opposite building standards\n"
     "5. Tsunami potential: submarine earthquakes displace water\n\n"
     "**Prediction**: no reliable short-term prediction exists. Long-term probabilistic forecasts (UCERF for California) estimate 63% chance of Mw 6.7+ in Bay Area within 30 years.",
     9.0, "Q451 Earthquakes: fault slip at tectonic boundaries; Mw log scale; destructiveness = depth+soil+buildings; subduction = largest; no reliable prediction"),

    (452, "Medicine", "What is antibiotic resistance and how serious is the threat?",
     "Antibiotic resistance occurs when bacteria evolve mechanisms to survive drugs that once killed them.\n\n"
     "**Mechanisms**:\n"
     "1. Enzyme production: beta-lactamases destroy penicillin ring\n"
     "2. Efflux pumps: bacteria actively expel antibiotics\n"
     "3. Target modification: MRSA alters penicillin-binding proteins\n"
     "4. Reduced permeability: gram-negatives limit drug entry\n\n"
     "**How resistance spreads**: horizontal gene transfer -- bacteria share resistance genes via plasmids (not just vertical inheritance). One resistant bacterium can share its resistance with unrelated species rapidly.\n\n"
     "**Scale of the problem**: WHO labels AMR a top-10 global health threat. 2019: ~1.27 million deaths directly attributable to AMR (Lancet). By 2050: projections of 10 million deaths/year (O'Neill Review 2016) if unchecked -- exceeding cancer.\n\n"
     "**Key culprits**: ESKAPE pathogens (Enterococcus, Staph aureus, Klebsiella, Acinetobacter, Pseudomonas, Enterobacter). CRE (carbapenem-resistant Enterobacteriaceae) resistant to last-resort antibiotics.\n\n"
     "**Drivers**: overprescribing in humans; massive agricultural use (70% of all antibiotics); incomplete courses; poor infection control.\n\n"
     "**Solutions**: new antibiotic development (dry pipeline -- low profit); phage therapy; antibiotic stewardship; vaccines reducing infection rates; rapid diagnostics.",
     9.5, "Q452 Antibiotic resistance: beta-lactamases, efflux pumps, target modification; horizontal gene transfer; 1.27M deaths/year now; ESKAPE pathogens; pipeline dry"),

    (453, "Psychology", "What is the bystander effect and why does it happen?",
     "The bystander effect: people are less likely to help in an emergency when others are present.\n\n"
     "**Origin**: Kitty Genovese (1964) -- stabbed outside her Queens apartment; 38 witnesses reportedly did nothing. Motivated Darley & Latane's classic experiments.\n\n"
     "**Experiment (1968)**: subject alone or with confederates hears someone having a seizure via intercom. Alone: 85% helped within 52 seconds. With 4 others: only 31% helped within 3 minutes.\n\n"
     "**Mechanisms**:\n"
     "1. **Diffusion of responsibility**: 'someone else will help' -- responsibility divided among observers\n"
     "2. **Pluralistic ignorance**: everyone looks calm so everyone assumes it's not an emergency so nobody acts\n"
     "3. **Evaluation apprehension**: fear of looking foolish if wrong about emergency\n"
     "4. **Ambiguity reduction**: unclear situations -- look to others -- see inaction -- assume no problem\n\n"
     "**Reversals**: bystander effect weakens when:\n"
     "- Emergency is unambiguous (blood, fire)\n"
     "- Bystander has relevant expertise\n"
     "- Victim is similar to helper\n"
     "- Direct eye contact with victim\n"
     "- You point at specific person: 'You in the red -- call 911'\n\n"
     "**Genovese note**: the '38 witnesses saw everything' story was exaggerated by media. Few had full view. Mechanism still real, origin story partly myth.",
     9.0, "Q453 Bystander effect (Darley & Latane 1968): diffusion of responsibility, pluralistic ignorance; 85% alone vs 31% with others; overcome by direct assignment"),

    (454, "Astronomy", "What is dark matter and what is the evidence for it?",
     "Dark matter: non-luminous mass that does not interact electromagnetically but does exert gravity. Makes up ~27% of universe's energy budget (ordinary matter ~5%, dark energy ~68%).\n\n"
     "**Evidence**:\n"
     "1. **Galaxy rotation curves** (Rubin 1970s): stars at galaxy edges orbit faster than Newtonian gravity from visible matter predicts. Flat rotation curve implies invisible mass halo\n"
     "2. **Galaxy clusters**: cluster mass from gravitational lensing far exceeds visible mass. Bullet Cluster (2006): collision of two clusters -- hot gas (visible) slowed by interaction; dark matter (lensing map) passed straight through\n"
     "3. **CMB**: acoustic peaks in cosmic microwave background fit perfectly only with ~27% dark matter component\n"
     "4. **Large-scale structure**: galaxies form along filaments consistent with cold dark matter simulations (LCDM)\n\n"
     "**What it might be**:\n"
     "- WIMPs (weakly interacting massive particles) -- theoretically motivated, null results at LHC and direct detection experiments (LUX, XENONnT)\n"
     "- Axions -- very light, QCD-motivated; ALP experiments ongoing\n"
     "- Primordial black holes -- LIGO constraints limit contribution\n"
     "- Sterile neutrinos\n\n"
     "**Alternative**: MOND (modified Newtonian dynamics) -- adjust gravity instead. Explains rotation curves but fails at cluster scales.\n\n"
     "**Status**: dark matter existence extremely well-supported; its particle nature unknown. A top unsolved problem in physics.",
     9.5, "Q454 Dark matter: 27% of universe; rotation curves, Bullet Cluster, CMB peaks as evidence; WIMPs/axions as candidates; particle nature unknown"),

    (455, "Literature", "What makes Shakespeare's work so enduringly significant?",
     "Shakespeare (1564-1616) wrote 37 plays, 154 sonnets, and long poems across comedy, tragedy, history, and romance -- in 20 years of active writing.\n\n"
     "**Language**: coined or first-recorded ~1,700 words (bedroom, lonely, generous, obscene, swagger). Blank verse (iambic pentameter) flexible enough for noble and colloquial speech alike. Puns, wordplay, metaphor density unprecedented.\n\n"
     "**Psychological depth**: characters reason, contradict themselves, change. Hamlet contemplates inaction; Macbeth understands his own corruption; Iago is evil and self-aware about it; Lear learns through suffering. Predates psychology as field by 300 years but maps inner life precisely.\n\n"
     "**Universality**: tragedies turn on jealousy (Othello), ambition (Macbeth), indecision (Hamlet), family (Lear) -- universal drives. No setting-specific politics that date the work.\n\n"
     "**Influence**: largest influence on English language after King James Bible. Freud used Hamlet to develop Oedipal theory. Harold Bloom argued Shakespeare 'invented the human' -- modern Western self-conception.\n\n"
     "**Survival bias note**: we have ~3,000 plays from Elizabethan era; most lost. Shakespeare's survived partly due to the First Folio (1623), compiled by actors who preserved 18 plays that would otherwise be lost.\n\n"
     "**Ongoing relevance**: performed in every country; adapted across cultures (Kurosawa's Ran = King Lear; 10 Things I Hate About You = Taming of the Shrew). The plays scale across contexts because they're about humans, not Tudor politics.",
     9.0, "Q455 Shakespeare: 37 plays, 1700 words coined; psychological depth predating psychology; Hamlet/Macbeth/Lear on universal drives; First Folio preserved 18 plays"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total: {total}")
