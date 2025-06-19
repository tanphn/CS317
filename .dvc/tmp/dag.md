```mermaid
flowchart TD
	node1["evaluate"]
	node2["preprocess"]
	node3["train"]
	node2-->node1
	node2-->node3
	node3-->node1
```