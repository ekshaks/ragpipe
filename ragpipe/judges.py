


def binary_judge(query, expected_response, response):
    prompt = f'''Given the following query, check if the response agrees with the expected response. Ignore spurious differences.
    If they agree, respond with 'Yes', otherwise respond with 'No'.  Do not generate any explanations or additional information.  

    query: {query}
    response: {response}
    expected response: {expected_response}'''

    from .llms import cloud_llm
    res = cloud_llm(prompt)
    if 'Yes' in res: return True
    else: return False

if __name__ == '__main__':
    query = "What is the capital of France?"
    expected_response = "Paris"
    response = "London"

    res = binary_judge(query, expected_response, response)
    print(res)