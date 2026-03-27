"""Save Q461-Q465 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (461, "Technology", "How does GPS work, and how accurate can it get?",
     "GPS (Global Positioning System): 31 satellites in medium Earth orbit (~20,200 km), each broadcasting its position and a precise timestamp from atomic clocks.\n\n"
     "**How it works**:\n"
     "1. Your receiver picks up signals from at least 4 satellites\n"
     "2. Signal travel time x speed of light = distance to each satellite\n"
     "3. 3 satellites give a 3D position (trilateration); 4th corrects receiver clock error\n"
     "4. Result: latitude, longitude, altitude, time\n\n"
     "**Accuracy**:\n"
     "- Standard civilian GPS: 3-5 metres horizontal\n"
     "- Differential GPS (DGPS): ground stations correct atmospheric error -- sub-metre\n"
     "- RTK (Real-Time Kinematic): carrier phase corrections -- centimetre accuracy. Used in surveying, autonomous vehicles\n"
     "- PPP (Precise Point Positioning): ~10cm without base station\n\n"
     "**Sources of error**:\n"
     "- Ionospheric delay: charged particles slow signals (~5m error; dual-frequency receivers compensate)\n"
     "- Tropospheric delay: water vapour (~1m)\n"
     "- Multipath: signals bouncing off buildings\n"
     "- Satellite geometry (GDOP): poorly positioned satellites amplify errors\n\n"
     "**Relativistic corrections**: both special relativity (satellites move fast -- clocks slow) and general relativity (gravity weaker at altitude -- clocks fast) must be corrected. Without it: 10km/day position error.\n\n"
     "**Other systems**: GLONASS (Russia), Galileo (EU), BeiDou (China) -- together called GNSS. Multi-constellation receivers are more accurate.",
     9.0, "Q461 GPS: trilateration from 4+ satellites; 3-5m standard, cm with RTK; ionospheric/multipath errors; requires relativistic corrections; GNSS = multi-constellation"),

    (462, "Cognitive Science", "What is consciousness and why is it so hard to explain?",
     "Consciousness is subjective experience -- 'what it is like' to be you. The hardest problem in science.\n\n"
     "**The Hard Problem** (Chalmers 1995): why does physical brain activity give rise to subjective experience (qualia)? We can explain attention, memory, behaviour (easy problems) -- but not why there is 'something it is like' to see red.\n\n"
     "**Major theories**:\n"
     "1. **Global Workspace Theory** (Baars, Dehaene): consciousness = information broadcast globally across brain. Explains reportability and access. Doesn't explain why broadcasting feels like anything.\n"
     "2. **Integrated Information Theory** (Tononi): consciousness = phi (integrated information). Any system with high phi is conscious. Implies possible consciousness in simple networks; disputed.\n"
     "3. **Higher-Order Theories**: you're conscious of X only when you have a thought about X. Explains metacognition.\n"
     "4. **Predictive Processing** (Clark, Friston): brain is a prediction machine; consciousness = models of the world and self.\n\n"
     "**Neural correlates**: fMRI shows prefrontal-parietal network active during conscious perception; thalamus as gating mechanism; gamma oscillations at 40Hz correlate with awareness.\n\n"
     "**Disorders of consciousness**: vegetative state patients sometimes show fMRI activity in response to commands -- covert awareness. Raises legal/ethical questions.\n\n"
     "**The mystery remains**: no theory explains why information processing feels like anything. Penrose/Hameroff: quantum effects in microtubules (mostly rejected). Panpsychism: consciousness is fundamental like mass or charge. Most neuroscientists remain physicalists but admit the explanation is missing.",
     9.5, "Q462 Consciousness: Hard Problem (Chalmers); GWT, IIT, HOT, predictive processing theories; NCC = prefrontal-parietal + gamma; no theory explains qualia yet"),

    (463, "Economics", "What caused the 2008 financial crisis, and did we fix it?",
     "Cascade of interconnected failures -- system designed to fail.\n\n"
     "Housing bubble: 2000-2006, US prices rose ~124%; fueled by low interest rates post-dotcom and predatory subprime lending. Banks lent to borrowers with no income, no assets (NINJA loans).\n\n"
     "**Securitisation machine**: banks packaged mortgages into MBS (mortgage-backed securities), sold to investors. Then CDOs (collateralised debt obligations) -- slices of MBS. Rating agencies stamped AAA on toxic tranches. No one held the risk.\n\n"
     "**Leverage**: banks leveraged 30:1+. Lehman Brothers: $30 of assets per $1 equity. 3% decline = insolvent.\n\n"
     "**Trigger**: 2006 peak, 2007 subprime defaults rise. Bear Stearns hedge funds collapse Aug 2007. Lehman Brothers bankrupt Sep 2008 -- $600B failure, largest in history. Money market funds 'broke the buck.' Credit seized globally.\n\n"
     "**Contagion**: derivatives (especially CDS -- credit default swaps) linked every major institution. AIG wrote $400B in CDS with no capital. Government bailed it out ($182B).\n\n"
     "**Response**: TARP ($700B bank bailout); QE ($3.5T asset purchases); Dodd-Frank 2010 (stress tests, Volcker rule, consumer protection).\n\n"
     "**Fixed?**: Banks are better capitalised (Basel III). But: shadow banking still large; leverage moved to private equity; systemic risk harder to see; political will to regulate fades in good times.",
     9.5, "Q463 2008 crisis: subprime mortgages -> MBS -> CDO securitisation; 30:1 leverage; Lehman $600B failure; CDS contagion; Dodd-Frank partial fix; systemic risk remains"),

    (464, "Linguistics", "How do children learn language so quickly -- and what does it tell us about the mind?",
     "Children go from babbling at 6 months to grammatical sentences at 3 years, mastering a system adults take years to learn deliberately. How?\n\n"
     "**Milestones**: cooing (2m) -> babbling (6m) -> first words (12m) -> two-word phrases (18-24m) -> sentences with grammar (36m). Critical period for native accent acquisition: before ~12 years.\n\n"
     "**Behaviorist view** (Skinner): imitation and reinforcement. Demolished by Chomsky (1959): children produce sentences they've never heard; they overgeneralise rules ('goed', 'mouses'); no amount of reinforcement explains this creativity.\n\n"
     "**Nativist view** (Chomsky): Language Acquisition Device (LAD) -- innate universal grammar. Children are pre-wired for language structure. Evidence: all languages share deep structural properties; deaf children invent grammar spontaneously; creoles have grammar.\n\n"
     "**Social-interactionist view** (Tomasello): children learn by joint attention and intention reading. They map words to what adults intend, not just what they point at. Very social species-specific.\n\n"
     "**Statistical learning** (Saffran): 8-month-olds can segment words from speech by tracking transitional probabilities between syllables -- no explicit teaching.\n\n"
     "**Poverty of stimulus** (Chomsky): input is too impoverished and noisy to explain what children learn. Some grammar must be innate.\n\n"
     "**Current consensus**: probably an interaction -- innate capacities (not full grammar, but learning biases and social cognition) + rich statistical learning + social scaffolding.",
     9.0, "Q464 Language acquisition: babbling-to-sentences by 3yr; Skinner demolished by Chomsky; nativist LAD vs social-interactionist vs statistical learning; poverty of stimulus"),

    (465, "History", "What was the Cold War and how did it shape the modern world?",
     "The Cold War (1947-1991): ideological, geopolitical, and military competition between the US-led West and Soviet-led East -- never direct war between superpowers, but constant proxy conflicts.\n\n"
     "**Origins**: post-WWII power vacuum. USSR occupied Eastern Europe; Truman Doctrine (1947) committed US to containing Soviet expansion; Marshall Plan ($13B) rebuilt Western Europe; NATO founded 1949; Soviets test A-bomb 1949.\n\n"
     "**Key crises**:\n"
     "- Korean War 1950-53: first proxy war; 3M dead\n"
     "- Berlin crises 1948, 1961 (Wall built)\n"
     "- Cuban Missile Crisis 1962: 13 days -- closest to nuclear war; Khrushchev/Kennedy back-channel deal\n"
     "- Vietnam 1955-75: US loses first war; 58k American, 2M+ Vietnamese dead\n"
     "- Soviet-Afghan War 1979-89: USSR's Vietnam\n\n"
     "**Arms race**: US/USSR built 70,000 nuclear warheads combined. MAD (Mutually Assured Destruction) -- deterrence through guaranteed annihilation. Accidentally stabilising.\n\n"
     "**Space race**: Sputnik 1957, Gagarin 1961, Apollo 11 1969. Drove technology (internet = ARPANET, GPS, microchips).\n\n"
     "**End**: Gorbachev's glasnost and perestroika; economic collapse of USSR; Berlin Wall falls Nov 1989; Soviet dissolution Dec 1991.\n\n"
     "**Legacy**: NATO expansion; US global hegemony; nuclear proliferation; Middle East interventions; China's rise; Russia's revanchism; authoritarian-democracy ideological divide persisting today.",
     9.5, "Q465 Cold War 1947-91: US vs USSR containment; Cuban Missile Crisis; MAD deterrence; Vietnam/Korea/Afghanistan proxies; Space Race -> technology; USSR collapse 1991"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total: {total}")
