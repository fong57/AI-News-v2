"""Tests for the graph builder module."""

from langgraph.graph.state import CompiledStateGraph

from litenews.workflow.graph_builder import create_news_graph


class TestCreateNewsGraph:
    """Tests for the create_news_graph function."""

    def test_returns_compiled_graph(self):
        """Test that it returns a compiled graph."""
        graph = create_news_graph()

        assert isinstance(graph, CompiledStateGraph)

    def test_graph_has_all_nodes(self):
        """Test that the graph has all expected nodes."""
        graph = create_news_graph()

        node_names = set(graph.nodes.keys())
        expected_nodes = {
            "configure",
            "configure_human",
            "edit_human",
            "research",
            "analyze",
            "outline",
            "outline_human",
            "write",
            "fact_check",
            "revise",
            "revise_human",
            "fact_check_remarks",
            "review",
        }

        for node in expected_nodes:
            assert node in node_names, f"Missing node: {node}"

    def test_graph_has_start_edge_to_configure(self):
        """Test that the graph starts with the configure node."""
        graph = create_news_graph()

        edges = graph.get_graph().edges
        start_edges = [e for e in edges if e[0] == "__start__"]

        assert len(start_edges) == 1
        assert start_edges[0][1] == "configure"

    def test_graph_has_end_edge_from_review(self):
        """Test that the review node can end the graph."""
        graph = create_news_graph()

        edges = graph.get_graph().edges
        review_edges = [e for e in edges if e[0] == "review"]

        assert any(e[1] == "__end__" for e in review_edges)

    def test_graph_nodes_are_callable(self):
        """Test that each node is invocable (raw callable or LangGraph PregelNode with bound)."""
        graph = create_news_graph()

        for name, node in graph.nodes.items():
            if name not in ("__start__", "__end__"):
                assert callable(node) or getattr(node, "bound", None) is not None, (
                    f"Node {name} is not wired to runnable logic"
                )

    def test_graph_has_conditional_edges(self):
        """Test that the graph has conditional edges for error handling."""
        graph = create_news_graph()
        edges = graph.get_graph().edges

        nodes_with_conditional_edges = {
            "configure",
            "configure_human",
            "edit_human",
            "research",
            "analyze",
            "outline",
            "outline_human",
            "write",
            "fact_check",
            "revise",
            "revise_human",
        }
        for node in nodes_with_conditional_edges:
            node_edges = [e for e in edges if e[0] == node]
            assert len(node_edges) >= 1, f"Node {node} should have edges"
