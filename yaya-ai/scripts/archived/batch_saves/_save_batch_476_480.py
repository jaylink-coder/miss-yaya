"""Save Q476-Q480 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (476, "Physics", "What is light and what is the photoelectric effect?",
     "Light is electromagnetic radiation -- oscillating electric and magnetic fields propagating through space at c (299,792,458 m/s in vacuum). It behaves as both wave and particle depending on how you measure it.\n\n"
     "**Wave properties**: wavelength determines colour (400nm violet to 700nm red); frequency determines energy (E=hf). Interference, diffraction, polarisation -- all wave behaviour.\n\n"
     "**Particle (photon) properties**: light comes in discrete packets (quanta). Each photon carries energy E=hf where h is Planck's constant.\n\n"
     "**Photoelectric effect**: when light hits a metal surface, electrons are ejected -- but only if frequency exceeds a threshold, regardless of intensity. Low-frequency bright light ejects no electrons; dim high-frequency light does.\n\n"
     "**Classical prediction**: higher intensity should always eject electrons by dumping more energy. This failed.\n\n"
     "**Einstein's explanation (1905, Nobel 1921)**: light comes in photons of energy E=hf. A single photon must have enough energy (high enough frequency) to overcome the electron's binding energy (work function). More photons (intensity) just eject more electrons -- they don't make each ejection easier.\n\n"
     "**Why it matters**:\n"
     "- Proved quantisation of light -- established quantum mechanics\n"
     "- Solar cells: photons knock electrons free in semiconductor, generating current (same principle)\n"
     "- Digital cameras: CCD sensors use photoelectric effect to convert light to electrical signal\n"
     "- Night vision, medical X-ray detectors, photomultiplier tubes\n\n"
     "**Wave-particle duality**: light is neither purely wave nor purely particle. It is a quantum object that exhibits both properties depending on experimental setup (double-slit: wave; photodetector: particle).",
     9.0, "Q476 Light/photoelectric effect: E=hf; threshold frequency not intensity; Einstein 1905 Nobel; proved photon quantisation; solar cells + CCDs use same principle"),

    (477, "History", "What was the Renaissance and why did it happen in Italy?",
     "Renaissance (French: 'rebirth'): European cultural, artistic, and intellectual revival from roughly 1300-1600, centred first in Italian city-states.\n\n"
     "**What it was**: revival of classical Greek and Roman learning; explosion of art, architecture, literature, science, philosophy. Shift from medieval theological focus to humanist celebration of human potential.\n\n"
     "**Key figures**: Leonardo da Vinci (art, engineering, anatomy, everything); Michelangelo (Sistine Chapel, David, Pieta); Raphael; Botticelli; Machiavelli (The Prince -- realpolitik); Petrarch (considered first humanist); Galileo (astronomy, scientific method).\n\n"
     "**Why Italy first?**\n"
     "1. **Wealth**: Italian city-states (Florence, Venice, Milan, Rome) were Europe's commercial centres. Banks (Medici), trade, textiles generated surplus for patronage.\n"
     "2. **City-state competition**: Medici vs Sforza vs Doge -- rival patrons competing through art and architecture. Florence alone had extraordinary concentration of talent.\n"
     "3. **Classical proximity**: Italy sat on Roman ruins. Greek manuscripts flooded in after fall of Constantinople 1453 -- Byzantine scholars fled west with texts.\n"
     "4. **Weak feudal system**: merchants, not hereditary nobility, dominated. More social mobility, more secular outlook.\n"
     "5. **Church patronage**: popes commissioned art (Sistine Chapel, St Peter's Basilica) -- enormous funding for artists.\n\n"
     "**Spread**: printing press (Gutenberg 1440) spread Renaissance ideas across Europe. Northern Renaissance: Erasmus, More, Shakespeare, Durer.\n\n"
     "**Legacy**: scientific revolution, Reformation, humanism -- the Renaissance ended the Middle Ages and began modernity.",
     9.0, "Q477 Renaissance 1300-1600: rebirth of classical learning; Italy first due to Medici wealth, city-state competition, 1453 Greek manuscripts; da Vinci, Michelangelo; printing spread it"),

    (478, "Neuroscience", "What is neuroplasticity and can the adult brain really change?",
     "Neuroplasticity: the brain's ability to reorganise itself by forming new neural connections throughout life. The old view -- brain fixed after childhood -- is wrong.\n\n"
     "**Types of plasticity**:\n"
     "1. **Synaptic plasticity**: individual synapse strength changes. Long-term potentiation (LTP) -- repeated firing strengthens connection ('neurons that fire together wire together', Hebb 1949). Basis of learning and memory.\n"
     "2. **Structural plasticity**: physical growth of new dendrites, synapses. Occurs in response to learning and experience.\n"
     "3. **Neurogenesis**: new neurons generated in adult brain -- primarily in hippocampus (dentate gyrus) and olfactory bulb. Controversial in humans but confirmed in rodents; human evidence growing.\n"
     "4. **Cortical remapping**: brain regions reassign to new functions after injury or deprivation.\n\n"
     "**Evidence**:\n"
     "- London taxi drivers: hippocampus posterior enlarged vs controls; size correlated with years of experience (Maguire 2000)\n"
     "- Blind individuals: visual cortex repurposed for tactile and auditory processing\n"
     "- Stroke recovery: neighbouring regions take over damaged functions with rehabilitation\n"
     "- Musicians: enlarged motor cortex representation for fingers; auditory cortex differences\n"
     "- Phantom limb: amputated limb representation invaded by face area (Ramachandran) -- explains why touching face triggers phantom sensation\n\n"
     "**Limits**: plasticity declines with age but never stops. Critical periods (language accent, binocular vision) close in childhood. Adult plasticity requires effort, attention, and repetition -- passive exposure doesn't rewire the brain.\n\n"
     "**Implications**: learning is physical change. Sleep consolidates plastic changes. Cognitive reserve from education delays dementia symptoms.",
     9.5, "Q478 Neuroplasticity: LTP (Hebb 1949), structural changes, neurogenesis; taxi drivers + musicians evidence; cortical remapping after injury; declines but never stops"),

    (479, "Philosophy", "What is the Ship of Theseus problem and why does it matter for identity?",
     "The Ship of Theseus: a ship's planks are replaced one by one over time. Eventually every original plank is replaced. Is it still the same ship?\n\n"
     "**Extended version** (Hobbes): if you collected all the original planks and reassembled them, which ship is the 'real' Ship of Theseus?\n\n"
     "**The problem**: it tests intuitions about identity, continuity, and what makes something the same thing over time.\n\n"
     "**Candidate criteria for identity**:\n"
     "1. **Material continuity**: same matter = same object. Fails -- our bodies replace most atoms every few years, yet we persist.\n"
     "2. **Structural continuity**: same form/structure = same object. Works better but: gradual vs sudden replacement matters?\n"
     "3. **Causal continuity**: the right kind of causal chain links past and present. The replacement ship descends causally from the original.\n"
     "4. **Functional continuity**: same function, same identity. The sailing ship is the same ship.\n"
     "5. **Psychological continuity** (for persons): Locke -- memory links personal identity. Derek Parfit: identity may not be what matters; what matters is psychological continuity and connectedness, even if identity is vague.\n\n"
     "**Real-world stakes**:\n"
     "- Personal identity: are you the same person as your 5-year-old self? (almost no shared neurons or memories)\n"
     "- Corporations: is Apple the same company that Steve Jobs founded?\n"
     "- Nations: is modern Greece the same as ancient Greece?\n"
     "- Medical: when does a patient in persistent vegetative state cease to be the same person?\n"
     "- AI: if an AI's weights are updated, is it the same system?\n\n"
     "**Parfit's conclusion**: personal identity is not what matters. What matters is psychological connectedness -- and that can hold to varying degrees. This undermines the moral importance we place on strict identity.",
     9.0, "Q479 Ship of Theseus: gradual replacement paradox; 5 identity criteria (material/structural/causal/functional/psychological); Parfit -- identity not what matters"),

    (480, "Sociology", "What is social stratification and why do inequalities persist?",
     "Social stratification: the hierarchical arrangement of individuals and groups in society based on wealth, power, prestige, and other resources.\n\n"
     "**Three dimensions** (Weber): class (economic), status (social honour/prestige), party (political power). These often align but can diverge (a respected professor may earn less than a plumber).\n\n"
     "**Major theoretical frameworks**:\n"
     "1. **Functionalism** (Davis & Moore 1945): inequality is functional -- it motivates talented people to fill important, demanding roles. Controversial and widely critiqued (ignores structural barriers; assumes meritocracy).\n"
     "2. **Conflict theory** (Marx): classes defined by relationship to means of production. Bourgeoisie exploit proletariat. Inequality is maintained by power, ideology, and force -- not merit.\n"
     "3. **Bourdieu**: multiple forms of capital -- economic, cultural (education, taste, credentials), social (networks), symbolic (prestige). Elite groups convert one into another, reproducing advantage across generations.\n\n"
     "**Why inequalities persist**:\n"
     "- **Intergenerational transmission**: wealth, education, networks, social capital inherited. Estate tax avoidance, legacy admissions, nepotism.\n"
     "- **Structural barriers**: neighbourhood schools reflect local tax base; zip code predicts life outcomes.\n"
     "- **Cultural reproduction** (Bourdieu): cultural capital (how you speak, what you know, tastes) signals class membership and gatekeeps elite institutions.\n"
     "- **Cumulative advantage**: Matthew effect -- 'to those who have, more will be given.' Small early advantages compound (head start in reading -> better school -> better job).\n"
     "- **Ideology**: belief in meritocracy legitimises inequality. People blame poverty on individual failure, not structure.\n\n"
     "**Mobility**: US has lower intergenerational mobility than most Western European countries despite 'American Dream' ideology (Chetty et al. 2014). Scandinavia has highest mobility.",
     9.0, "Q480 Social stratification: Weber's 3 dimensions; Davis-Moore functionalism vs Marx conflict vs Bourdieu capital; Matthew effect; US lower mobility than Scandinavia"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total: {total}")
