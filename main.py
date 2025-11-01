def main():
    print("Hello from ramesana!")

    from vllm import LLM, SamplingParams
    prompts = [
        "Write a friendly, respectful email to your favorite person",
        "The largest country in the world is ",
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(model="facebook/opt-125m", dtype="float16")
    responses = llm.generate(prompts, sampling_params)

    for response in responses:
        print(response.outputs[0].text)



if __name__ == "__main__":
    main()
