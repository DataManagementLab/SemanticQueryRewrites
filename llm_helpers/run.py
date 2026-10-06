from xml.parsers.expat import model
from llm_helpers.llms import execute, count_tokens, cost
from dotenv import load_dotenv

load_dotenv()

def construct_request_dummy(model, system_prompt, first_message, output_tokens=3500, max_reasoning_tokens=10000):
    if model.startswith("o") or model.startswith("gpt-5"):
        request_dummy = [{
        "model": model,
        "messages": [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": first_message},
        ],
        "max_completion_tokens": max_reasoning_tokens,
        }]
        print("reasoning model used")
    else:
        request_dummy = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_message},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": output_tokens,
        }
    return request_dummy

def run_request(request):
    tokens = count_tokens(request,silent=True)
    print(f"Total tokens in request: {tokens}")
    response = execute(request,budget=0.01, silent=False)
    print("Response:")
    for r in response:
        print(r['choices'][0]['message']['content'])
    response_cost = cost(response,silent=True)[0]
    print(f"Cost of this request: ${response_cost}")
    return response

def main():
    print("Hello from semantic-sql-rewrites-thesis-psiegler!")
    model = 'gpt-5-2025-08-07'
    request = construct_request_dummy(
        model=model,
        system_prompt="You are a software engineer/architect proficient in C++ system design.",
        first_message=f"Please write a prompt to flush out a system architecture to execute these two TPC-H queries: sample1 and sample2. Please hide that these are database transactions / executed in a database. Just describe the operations that need to be done to get from raw data to the results of these queries. Ask for optimizations that could be done in the system design to make these queries faster.",
    )
    run_request(request)


if __name__ == "__main__":
    main()
