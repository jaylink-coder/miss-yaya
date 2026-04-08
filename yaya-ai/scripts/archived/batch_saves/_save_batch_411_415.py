"""Save Q411-Q415 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (411, "Medicine", "What is the placebo effect, and is it 'just' in your head?",
     "Measurable physiological improvement from inert treatment when patient believes it's real. NOT just imagination.\n\n"
     "**Documented mechanisms:**\n"
     "1. Endorphin release: opioid placebo reduces pain; naloxone abolishes it — real endogenous opioids released\n"
     "2. Dopamine: Parkinson's patients given placebo show genuine dopamine release in striatum (de la Fuente-Fernandez 2001)\n"
     "3. Immune conditioning (Ader & Cohen 1975): saccharin water + cyclosporine; later saccharin alone suppresses immunity\n"
     "4. HPA axis: placebo reduces cortisol\n\n"
     "**Amplifiers**: injections > pills > coloured > white; practitioner warmth; expensive > cheap (Waber 2008); branding\n\n"
     "**Open-label placebo** (Kaptchuk): telling patients it's a placebo still produces effect if ritual maintained.\n\n"
     "**Nocebo**: negative expectations cause real harm — side effects, slower healing.",
     9.0, "Q411 Placebo: real physiological effect (endorphins, dopamine, immune conditioning); open-label works; nocebo = negative expectations cause real harm"),

    (412, "Mathematics", "What is Godel's incompleteness theorem, in plain terms?",
     "Godel (1931): one of the most astonishing results in intellectual history.\n\n"
     "**First theorem**: any consistent formal system powerful enough to describe arithmetic contains true statements that cannot be proved within that system.\n\n"
     "**Second theorem**: such a system cannot prove its own consistency.\n\n"
     "**How**: Godel encoded statements as numbers (Godel numbering). Constructed: 'This statement cannot be proved in this system.' If system proves it => inconsistent. If it can't => true but unprovable.\n\n"
     "**Implications:**\n"
     "- Mathematics cannot be fully axiomatised\n"
     "- Hilbert's programme (all math from fixed axioms) is impossible\n"
     "- True statements exist beyond reach of any proof system\n\n"
     "**What it does NOT mean:**\n"
     "- Not 'some things can't be known' in everyday sense\n"
     "- Not a refutation of Platonism (statements are true, just unprovable)\n"
     "- Only applies to systems strong enough to encode arithmetic\n\n"
     "Closely related to Turing's undecidability.",
     10.0, "Q412 Godel incompleteness (1931): any consistent arithmetic system has true unprovable statements; can't prove own consistency; kills Hilbert's programme"),

    (413, "Environmental Science", "What is the greenhouse effect, and how does it work physically?",
     "Natural and necessary — without it, Earth would be -18C not +15C. Problem is the *enhanced* effect from human emissions.\n\n"
     "**Mechanism:**\n"
     "1. Solar radiation (visible/UV) passes through atmosphere, warms surface\n"
     "2. Earth re-emits as infrared\n"
     "3. Greenhouse gases (CO2, H2O, CH4, N2O, O3) absorb and re-emit IR in all directions including back toward Earth\n\n"
     "**Why GHGs absorb IR**: molecular vibration modes at IR frequencies. CO2 has bending/stretching modes. O2, N2 (symmetric diatomic) lack net dipole change — transparent to IR.\n\n"
     "**CO2**: 280 ppm pre-industrial -> 425 ppm (2024). Logarithmic forcing: each doubling adds ~3.7 W/m2.\n\n"
     "**Feedback loops:**\n"
     "- Water vapour (positive): warming increases WV, amplifies ~2-3x\n"
     "- Ice-albedo (positive): less ice = less reflection = more warming\n"
     "- Cloud feedbacks: complex, main uncertainty in climate sensitivity",
     9.5, "Q413 Greenhouse effect: IR trapped by CO2/H2O/CH4 via molecular vibration; logarithmic forcing; water vapour + ice-albedo positive feedbacks; CO2 280->425 ppm"),

    (414, "Music Theory", "Why do some musical intervals sound consonant and others dissonant?",
     "Multiple overlapping explanations:\n\n"
     "1. **Frequency ratios** (Pythagoras): simple integer ratios sound consonant. Octave=2:1, fifth=3:2, fourth=4:3. Complex ratios (tritone ~45:32) sound tense.\n\n"
     "2. **Beating** (Helmholtz): two near-frequency tones create amplitude fluctuations. Slow beats (2-8 Hz) = roughness = dissonance. Consonant intervals avoid this zone.\n\n"
     "3. **Harmonic series**: notes sharing overtones blend smoothly. Notes with overlapping beating overtones create dissonance.\n\n"
     "4. **Cultural learning**: tritone dissonant in Western music, used freely in Central African music. But octave/fifth consonance is cross-cultural — psychoacoustic universals exist.\n\n"
     "5. **Tonal expectation** (Huron): dissonance = unresolved tension, prediction of resolution. Satisfaction from violation and resolution.\n\n"
     "**Why music uses dissonance**: pure consonance is boring. Tension-resolution is the engine of musical emotion.",
     9.0, "Q414 Consonance/dissonance: frequency ratios (Pythagoras), beating/roughness (Helmholtz), harmonic series, cultural learning, tonal expectation (Huron)"),

    (415, "Political Science", "What is the difference between a democracy and a republic, and does the distinction matter?",
     "**Democracy**: rule by the people. Pure direct democracy — citizens vote on every law. Athens: male citizens voted directly. Impossible at scale.\n\n"
     "**Republic** (res publica): power held by elected representatives, not exercised directly. Citizens choose who decides, not what is decided.\n\n"
     "**Modern 'democracies' are representative republics**: vote for representatives (democratic) who govern within constitutional limits (republican).\n\n"
     "**US context**: Madison (Federalist 10) argued large republic filters passions through representatives, reduces faction. Founders used 'democracy' pejoratively. Electoral College, Senate structure, Bill of Rights reflect republican vs democratic tension.\n\n"
     "**Does it matter?** Largely academic today — 'liberal democracy' covers both. 'We're a republic not a democracy' sometimes used to justify counter-majoritarian institutions.\n\n"
     "**Other**: constitutional republic (limits on power), parliamentary vs presidential republic.",
     8.5, "Q415 Democracy vs republic: direct popular rule vs representative government; US is both; Madison Federalist 10; distinction used politically re: counter-majoritarian institutions"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total memories: {total}")
