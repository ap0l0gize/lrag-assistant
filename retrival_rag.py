"""Legacy CLI — delegates to pipelines.baseline_vector."""

from pipelines.baseline_vector import generate_answer, generate_answer_detailed, get_context_and_preview

__all__ = ["generate_answer", "generate_answer_detailed", "get_context_and_preview"]

if __name__ == "__main__":
    test_question = "Kiedy rozpoczyna się i kończy rekrutacja na pierwszy rok studiów?"
    print("Agent analyzes your question and prepares a response...")
    answer = generate_answer(test_question, verbose=True)
    print("\n--- GENERATED RESPONSE ---")
    print(answer)
