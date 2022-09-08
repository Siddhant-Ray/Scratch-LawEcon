# Summary of labour contracts

In this task, I build a relational graph between entities extracted from a Labour Union corpus, using the Relatio package. The final relational graph is a directed graph, built on top of the Python networkx package.

## File structure

- Loading: The [load_contracts.py](load_contracts.py) file fetches the data and pre-processes it to a form to pass to Relatio for NER extraction.
- Generation: The [run_relatio.py](run_relatio.py) file runs the Relatio extraction pipeline and builds the directed graph.
