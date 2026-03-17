"""Generate synthetic training data for end-to-end pipeline validation.

Creates a small corpus of diverse text for testing the full pipeline:
tokenizer training -> data processing -> model training -> evaluation -> generation.

Usage:
    python scripts/generate_test_data.py --output_dir data/test
"""

import argparse
import os
import random

# Diverse text templates for synthetic corpus
TOPICS = {
    "science": [
        "The process of photosynthesis converts sunlight into chemical energy. Plants use chlorophyll in their leaves to absorb light, primarily in the blue and red wavelengths. This energy drives the conversion of carbon dioxide and water into glucose and oxygen. The overall equation is: 6CO2 + 6H2O + light -> C6H12O6 + 6O2.",
        "Gravity is one of the four fundamental forces of nature. According to Einstein's general theory of relativity, gravity is not a force but a curvature of spacetime caused by mass and energy. Objects follow geodesics through this curved spacetime, which we perceive as gravitational attraction.",
        "DNA replication is a semiconservative process where each strand of the double helix serves as a template for a new complementary strand. The enzyme helicase unwinds the double helix, while DNA polymerase adds nucleotides to the growing strand. This ensures genetic information is faithfully copied during cell division.",
        "The periodic table organizes elements by their atomic number and chemical properties. Elements in the same group share similar chemical behavior because they have the same number of electrons in their outer shell. Mendeleev first published a periodic table in 1869, predicting the existence of undiscovered elements.",
        "Quantum mechanics describes the behavior of matter at atomic and subatomic scales. The Heisenberg uncertainty principle states that we cannot simultaneously know both the exact position and momentum of a particle. This fundamental limit is not due to measurement imprecision but is an inherent property of nature.",
        "Evolution by natural selection is the process by which organisms with favorable traits are more likely to survive and reproduce. Over many generations, this leads to changes in the inherited characteristics of populations. Charles Darwin and Alfred Russel Wallace independently proposed this mechanism.",
        "The human brain contains approximately 86 billion neurons, each connected to thousands of other neurons through synapses. Neural signals are transmitted through a combination of electrical impulses and chemical neurotransmitters. This complex network gives rise to consciousness, memory, and thought.",
        "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape. They form when massive stars collapse at the end of their life cycle. The boundary beyond which escape is impossible is called the event horizon.",
    ],
    "technology": [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed. Supervised learning uses labeled examples, while unsupervised learning finds patterns in unlabeled data. Deep learning uses neural networks with many layers to learn hierarchical representations.",
        "The internet operates through a system of interconnected networks using standardized communication protocols. TCP/IP provides reliable, ordered delivery of data packets between applications. HTTP, built on top of TCP, enables the World Wide Web by defining how messages are formatted and transmitted.",
        "Cryptography secures digital communication through mathematical algorithms. Public key cryptography uses a pair of keys: a public key for encryption and a private key for decryption. RSA, one of the first public key systems, relies on the computational difficulty of factoring large prime numbers.",
        "Cloud computing provides on-demand access to computing resources over the internet. Infrastructure as a Service provides virtual machines and storage, Platform as a Service provides development environments, and Software as a Service provides complete applications. Major providers include AWS, Azure, and Google Cloud.",
        "Transformers are a neural network architecture that uses self-attention mechanisms to process sequential data. Unlike recurrent networks, transformers can process all positions in parallel, making them much faster to train. They have become the foundation for modern language models and many other AI systems.",
        "Containerization using Docker packages applications with their dependencies into portable units. Kubernetes orchestrates these containers across clusters of machines, handling scaling, networking, and fault tolerance. This approach has revolutionized software deployment and microservice architectures.",
        "Version control systems like Git track changes to source code over time. Branches allow parallel development, while merging combines different lines of work. Distributed version control means every developer has a complete copy of the repository history.",
        "Databases store and organize data for efficient retrieval. Relational databases use SQL and enforce structured schemas, while NoSQL databases offer flexible schemas for unstructured data. The choice depends on the application's consistency, availability, and partition tolerance requirements.",
    ],
    "history": [
        "The Industrial Revolution, beginning in the late 18th century in Britain, transformed economies from agrarian to manufacturing-based. Steam power, mechanized textile production, and iron smelting were key innovations. This period fundamentally changed social structures, urbanization patterns, and global trade.",
        "The Renaissance was a cultural movement that began in Italy in the 14th century and spread throughout Europe. It marked a renewed interest in classical Greek and Roman learning. Artists like Leonardo da Vinci and Michelangelo created masterworks, while thinkers like Galileo advanced scientific understanding.",
        "The invention of the printing press by Johannes Gutenberg around 1440 revolutionized the spread of information. Before printing, books were hand-copied by scribes, making them rare and expensive. The printing press made knowledge accessible to a much wider audience and accelerated the pace of intellectual change.",
        "The French Revolution of 1789 overthrew the monarchy and established principles of citizenship and inalienable rights. The Declaration of the Rights of Man proclaimed liberty, equality, and fraternity. The revolution's ideals influenced democratic movements worldwide for centuries to come.",
        "Ancient Egypt flourished along the Nile River for over three thousand years. The civilization built monumental pyramids, developed hieroglyphic writing, and made advances in medicine, mathematics, and astronomy. The annual flooding of the Nile provided fertile soil for agriculture.",
        "The Silk Road was a network of trade routes connecting East Asia to the Mediterranean. For over a thousand years, merchants exchanged silk, spices, precious metals, and ideas along these paths. The cultural exchange facilitated by the Silk Road shaped civilizations across Eurasia.",
    ],
    "mathematics": [
        "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides. Written as a squared plus b squared equals c squared, this relationship has been proven in hundreds of different ways and is fundamental to geometry.",
        "Calculus, developed independently by Newton and Leibniz in the 17th century, provides tools for analyzing change and accumulation. Differential calculus deals with rates of change through derivatives, while integral calculus deals with accumulation through integrals. Together they form the foundation of modern science and engineering.",
        "Prime numbers are natural numbers greater than one that have no positive divisors other than one and themselves. The fundamental theorem of arithmetic states that every integer greater than one can be uniquely expressed as a product of primes. The distribution of primes remains one of mathematics' deepest mysteries.",
        "Linear algebra studies vector spaces and linear transformations between them. Matrices represent these transformations and enable efficient computation. Eigenvalues and eigenvectors reveal fundamental properties of transformations and appear throughout physics, engineering, and data science.",
        "Probability theory provides a mathematical framework for quantifying uncertainty. Bayes' theorem describes how to update beliefs given new evidence. The central limit theorem states that the sum of many independent random variables tends toward a normal distribution, regardless of their individual distributions.",
        "Set theory, developed by Georg Cantor, provides the foundation for modern mathematics. Cantor showed that there are different sizes of infinity: the set of real numbers is uncountably infinite, while the set of natural numbers is countably infinite. This revolutionary idea transformed mathematical thinking.",
    ],
    "conversation": [
        "User: What is the capital of France?\nAssistant: The capital of France is Paris. It is located in the north-central part of the country along the Seine River. Paris is known for landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.",
        "User: How do I make a cup of tea?\nAssistant: To make a cup of tea, start by boiling fresh water. Place a tea bag or loose leaf tea in your cup. Pour the hot water over the tea and let it steep for three to five minutes, depending on how strong you like it. Remove the tea bag and add milk or honey if desired.",
        "User: Explain recursion in programming.\nAssistant: Recursion is when a function calls itself to solve a problem by breaking it into smaller subproblems. Each recursive call works on a simpler version of the original problem. A base case stops the recursion to prevent infinite loops. A classic example is calculating factorials: factorial of n equals n times factorial of n minus one, with factorial of zero equaling one.",
        "User: What are the benefits of exercise?\nAssistant: Regular exercise offers numerous health benefits. It strengthens the cardiovascular system, improves muscle tone and bone density, and helps maintain a healthy weight. Exercise also reduces stress, improves sleep quality, and boosts mental health by releasing endorphins. Even moderate activity like daily walking can have significant positive effects.",
        "User: Tell me about the solar system.\nAssistant: Our solar system consists of the Sun and everything gravitationally bound to it. There are eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. The inner planets are rocky, while the outer planets are gas or ice giants. The system also includes dwarf planets like Pluto, asteroids, comets, and countless smaller objects.",
        "User: What causes rainbows?\nAssistant: Rainbows form when sunlight enters water droplets in the atmosphere. The light refracts as it enters the droplet, reflects off the back surface, and refracts again as it exits. Different wavelengths of light refract at slightly different angles, separating white light into its component colors: red, orange, yellow, green, blue, indigo, and violet.",
    ],
}


def generate_corpus(output_dir: str, num_docs: int = 500, seed: int = 42):
    """Generate a synthetic text corpus for testing."""
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    all_texts = []
    for topic, paragraphs in TOPICS.items():
        for para in paragraphs:
            all_texts.append(para)

    # Generate documents by combining paragraphs
    documents = []
    for i in range(num_docs):
        num_paragraphs = random.randint(1, 4)
        selected = random.choices(all_texts, k=num_paragraphs)
        doc = "\n\n".join(selected)
        documents.append(doc)

    # Write raw text file for tokenizer training
    raw_path = os.path.join(output_dir, "corpus.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc + "\n\n")

    # Write individual text files for data processing
    docs_dir = os.path.join(output_dir, "raw")
    os.makedirs(docs_dir, exist_ok=True)

    # Split into a few files
    chunk_size = len(documents) // 5
    for i in range(5):
        start = i * chunk_size
        end = start + chunk_size if i < 4 else len(documents)
        chunk_path = os.path.join(docs_dir, f"chunk_{i:03d}.txt")
        with open(chunk_path, "w", encoding="utf-8") as f:
            for doc in documents[start:end]:
                f.write(doc + "\n\n")

    # Write eval data (smaller)
    eval_dir = os.path.join(output_dir, "eval_raw")
    os.makedirs(eval_dir, exist_ok=True)
    eval_docs = random.choices(all_texts, k=50)
    eval_path = os.path.join(eval_dir, "eval.txt")
    with open(eval_path, "w", encoding="utf-8") as f:
        for doc in eval_docs:
            f.write(doc + "\n\n")

    total_chars = sum(len(d) for d in documents)
    print(f"Generated synthetic corpus:")
    print(f"  Documents: {len(documents)}")
    print(f"  Topics: {list(TOPICS.keys())}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Raw corpus: {raw_path}")
    print(f"  Chunked files: {docs_dir}/ (5 files)")
    print(f"  Eval data: {eval_path}")

    return raw_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test data")
    parser.add_argument("--output_dir", type=str, default="data/test")
    parser.add_argument("--num_docs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_corpus(args.output_dir, args.num_docs, args.seed)


if __name__ == "__main__":
    main()
