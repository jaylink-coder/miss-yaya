"""Save Q481-Q485 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (481, "Computer Science", "What is machine learning and how does a neural network actually learn?",
     "Machine learning: systems that improve their performance on a task through experience, without being explicitly programmed with rules.\n\n"
     "**Three paradigms**:\n"
     "1. **Supervised**: learn from labelled examples (input -> correct output). Classification, regression.\n"
     "2. **Unsupervised**: find patterns in unlabelled data. Clustering, dimensionality reduction.\n"
     "3. **Reinforcement**: agent takes actions, receives rewards, learns policy maximising cumulative reward.\n\n"
     "**How a neural network learns**:\n"
     "1. **Forward pass**: input flows through layers of neurons. Each neuron: weighted sum of inputs + bias, passed through activation function (ReLU, sigmoid). Output is a prediction.\n"
     "2. **Loss**: compare prediction to ground truth using a loss function (cross-entropy for classification, MSE for regression).\n"
     "3. **Backpropagation**: compute gradient of loss with respect to every weight -- using chain rule of calculus. How much did each weight contribute to the error?\n"
     "4. **Gradient descent**: update each weight in the direction that reduces loss: w = w - lr * gradient. Learning rate controls step size.\n"
     "5. **Repeat**: billions of times over the training data (stochastic gradient descent: mini-batches).\n\n"
     "**What networks learn**: not rules. Distributed representations. Early layers detect edges/colours; middle layers detect shapes; later layers detect objects. Hierarchy of features, emergent from optimisation.\n\n"
     "**Key innovations**: ReLU (avoids vanishing gradient), dropout (prevents overfitting), batch normalisation (stabilises training), attention mechanism (transformers -- learns which parts of input to focus on).\n\n"
     "**Scale matters**: more parameters + more data + more compute = better performance. GPT-4 has ~1 trillion parameters; trained on internet-scale text.",
     9.5, "Q481 Neural network learning: forward pass, loss, backprop (chain rule), gradient descent; distributed representations; ReLU/dropout/attention; scale = performance"),

    (482, "Ethics", "What is effective altruism and what are the strongest criticisms of it?",
     "Effective altruism (EA): a philosophical and social movement using evidence and reason to determine the most effective ways to benefit others.\n\n"
     "**Core ideas** (Peter Singer, William MacAskill):\n"
     "- Cause neutrality: don't assume your local community or country matters more\n"
     "- Impartiality: future people and distant strangers count equally\n"
     "- Evidence-based: measure impact rigorously (GiveWell: cost per life saved)\n"
     "- Earning to give: high-salary career + donating effectively can do more good than low-pay NGO work\n"
     "- Longtermism: existential risks (AI, pandemics, nuclear) affect vastly more future people -- may outweigh all present suffering\n\n"
     "**Top EA causes**: global health (malaria nets, deworming -- GiveDirectly, Against Malaria Foundation); animal welfare (factory farming causes immense suffering at scale); AI safety (preventing catastrophic AI risk).\n\n"
     "**Strongest criticisms**:\n"
     "1. **Systemic change neglect**: EA focuses on charity, not structural causes of poverty. Fixes symptoms, not systems (Amia Srinivasan).\n"
     "2. **Legibility bias**: only measurable impacts count. Hard-to-measure things (art, culture, community, dignity) get ignored.\n"
     "3. **Longtermism hubris**: speculative future trillions outweigh present suffering -- used to justify ignoring current needs for uncertain future benefits.\n"
     "4. **Demandingness**: Singer's argument (drowning child) implies most people should give until marginal utility of their money equals marginal utility of recipient -- radical implication most reject.\n"
     "5. **FTX scandal** (2022): Sam Bankman-Fried, prominent EA figure, committed massive fraud 'for the greater good' -- exposed utilitarian ends-justify-means risk.\n\n"
     "**Response from EA**: criticisms taken seriously; movement has evolved. Systemic change and policy increasingly included.",
     9.0, "Q482 Effective altruism: cause neutrality, evidence-based, longtermism (MacAskill/Singer); criticisms: systemic neglect, legibility bias, FTX scandal, demandingness"),

    (483, "Chemistry", "How do batteries work, and what are the limits of lithium-ion?",
     "A battery converts chemical energy to electrical energy through controlled redox (reduction-oxidation) reactions.\n\n"
     "**Basic principle**: two electrodes (anode and cathode) separated by electrolyte. At anode: oxidation -- atoms lose electrons. At cathode: reduction -- atoms gain electrons. Electrons flow through external circuit (that's the current). Ions flow through electrolyte to balance charge.\n\n"
     "**Lithium-ion (dominant technology)**:\n"
     "- Anode: graphite (lithium ions intercalate between carbon layers during charging)\n"
     "- Cathode: lithium metal oxide (LiCoO2, LiFePO4, NMC variants)\n"
     "- Electrolyte: lithium salt in organic solvent\n"
     "- Charging: external current forces Li+ from cathode to anode; discharging reverses\n"
     "- Energy density: ~250 Wh/kg (much better than lead-acid at 35 Wh/kg)\n\n"
     "**Why lithium?**: lightest metal, high electrochemical potential, small ion fits in electrode lattices.\n\n"
     "**Limits of lithium-ion**:\n"
     "1. **Energy density ceiling**: theoretical max ~350 Wh/kg -- approaching it. Not enough for long-haul aviation or shipping.\n"
     "2. **Dendrite growth**: lithium metal anodes (better density) form needle-like dendrites that pierce separator -- short circuit, fire.\n"
     "3. **Thermal runaway**: puncture, overcharge, or manufacturing defect can cascade to fire. Aviation bans bulk Li shipment.\n"
     "4. **Cycle degradation**: capacity fades after 500-1000 charge cycles. SEI (solid electrolyte interphase) layer grows.\n"
     "5. **Supply chain**: cobalt (DRC, ethical concerns), lithium (Chile/Australia), nickel -- geopolitically concentrated.\n\n"
     "**Next generation**: solid-state batteries (solid electrolyte -- no fire risk, higher density, faster charge); lithium-sulphur (theoretical 2600 Wh/kg); sodium-ion (cheaper, abundant). All face commercialisation challenges.",
     9.0, "Q483 Batteries: redox reactions; Li-ion = graphite/LiCoO2/electrolyte, 250 Wh/kg; limits = dendrites, thermal runaway, cycle fade, cobalt supply; solid-state next"),

    (484, "Astronomy", "How did the universe begin, and what happened in the first seconds?",
     "The Big Bang: the universe began ~13.8 billion years ago from an extremely hot, dense state. Not an explosion in space -- an expansion of space itself.\n\n"
     "**Evidence**:\n"
     "1. **Hubble expansion** (1929): galaxies receding -- further away, faster recession. Run time backward: convergence.\n"
     "2. **CMB** (Cosmic Microwave Background): Penzias & Wilson 1965 -- faint 2.7K microwave radiation in every direction. Predicted by Big Bang theory as afterglow of recombination.\n"
     "3. **Light element abundances**: Big Bang nucleosynthesis predicts ~75% hydrogen, ~25% helium, trace lithium. Observed match is precise.\n\n"
     "**First seconds** (timeline):\n"
     "- t=10^-43s (Planck time): physics breaks down. Quantum gravity needed.\n"
     "- t=10^-36s: inflation -- exponential expansion (10^26 in 10^-33s). Explains flatness and horizon problems.\n"
     "- t=10^-12s: electroweak phase transition -- weak and electromagnetic forces separate.\n"
     "- t=10^-6s: quarks confine into protons and neutrons (quark-hadron transition).\n"
     "- t=1s: neutrinos decouple; electron-positron annihilation heats photons.\n"
     "- t=3min: nucleosynthesis -- protons and neutrons fuse into helium nuclei. Stops when universe cools too much.\n"
     "- t=380,000 years: recombination -- electrons combine with nuclei; universe becomes transparent. CMB emitted.\n"
     "- t=400M years: first stars ignite (Population III stars -- pure hydrogen/helium, massive, short-lived).\n\n"
     "**What came before?**: unknown. Hawking: the question may be meaningless -- 'before' requires time, which began with the Big Bang. Some models: eternal inflation, cyclic universes, quantum fluctuation from nothing.",
     10.0, "Q484 Big Bang: 13.8Gyr, expansion not explosion; Hubble+CMB+nucleosynthesis evidence; inflation at 10^-36s; quark confinement, nucleosynthesis, recombination timeline"),

    (485, "Psychology", "What is cognitive dissonance and how do people resolve it?",
     "Cognitive dissonance (Festinger 1957): the mental discomfort that arises when a person holds two or more contradictory beliefs, values, or when their actions contradict their beliefs.\n\n"
     "**Classic experiment** (Festinger & Carlsmith 1959): subjects did a boring task, then told the next participant it was interesting -- for $1 or $20. Those paid $1 rated the task as more enjoyable. Why? $20 justified the lie; $1 didn't -- so the $1 group changed their belief to reduce dissonance.\n\n"
     "**How people resolve dissonance**:\n"
     "1. **Change the belief**: smoker learns smoking causes cancer -> quits. Ideal but hardest.\n"
     "2. **Change the behaviour**: actually quits. Also ideal.\n"
     "3. **Add consonant cognitions**: 'I know smokers who lived to 100'; 'Stress kills too'; 'I'll quit eventually.' Easier but doesn't solve the problem.\n"
     "4. **Reduce importance**: 'Health isn't everything.' Minimise the dissonant element.\n"
     "5. **Denial**: ignore or reject the contradictory evidence. Most common under threat.\n\n"
     "**Real-world examples**:\n"
     "- Cult members whose prophecy fails: often double down rather than leave (Festinger's 'When Prophecy Fails' 1956)\n"
     "- Hazing: 'I suffered to join, so this group must be valuable' (justification of effort)\n"
     "- Consumer psychology: post-purchase rationalisation -- we believe we made a good choice to reduce doubt\n"
     "- Political beliefs: voters ignore evidence contradicting their candidate\n\n"
     "**Self-perception theory** (Bem): alternative explanation -- we infer our beliefs from observing our behaviour, not necessarily from resolving discomfort. Both theories explain some results better.\n\n"
     "**Why it matters**: understanding dissonance explains why people are resistant to changing minds even with evidence. The threat model: contradicting a core belief is perceived as an attack on the self.",
     9.0, "Q485 Cognitive dissonance (Festinger 1957): contradictory beliefs cause discomfort; $1 vs $20 experiment; 5 resolution strategies; cult doubling-down; self-perception alternative"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total: {total}")
