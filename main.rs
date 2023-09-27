pub struct Chatbot {
    model: Box<dyn llm_base::Model>,
    inference_params: llm::InferenceParameters,
    stop_sequence: String,
}

impl Chatbot {
    fn new() -> Self {
        let model_architecture = llm::ModelArchitecture::Gpt2; 
        let model_path = std::path::PathBuf::from("1SdjAt39Mjfi2mklO.bin"); 
        let sampler_string = format!("repetition:last_n=4:penalty=1.13/topk:k=1/topp:p=0.6/temperature:temperature=0.22");

        let model = llm::load_dynamic(
            Some(model_architecture),
            &model_path,
            llm::TokenizerSource::Embedded,
            llm::ModelParameters {
                prefer_mmap: false,
                context_size: 1300,
                lora_adapters: None,
                use_gpu: false,
                gpu_layers: None,
                rope_overrides: None,
                n_gqa: None,
            },
            llm::load_progress_callback_stdout,
        )
        .unwrap_or_else(|err| {
            panic!("Failed to load {:?} model from {:?}: {}", model_architecture, model_path, err)
        });

        let sampler = llm_base::samplers::build_sampler(0, Default::default(), &[sampler_string])
            .unwrap_or_else(|err| panic!("Failed to build sampler: {}", err));

        let inference_params = llm::InferenceParameters { sampler };
        let stop_sequence = "<".to_string();

        Chatbot {
            model,
            inference_params,
            stop_sequence,
        }
    }

fn process_user_input(&self, prompt: &str) -> String {
    let mut session: llm_base::InferenceSession = self.model.start_session(llm::InferenceSessionConfig {
        memory_k_type: llm::ModelKVMemoryType::Float16,
        memory_v_type: llm::ModelKVMemoryType::Float16,
        n_batch: 8,
        n_threads: 4,
    });

    let mut generated_text = String::new();  

    let _res = session.infer::<std::convert::Infallible>(
        self.model.as_ref(),
        &mut rand::thread_rng(),
        &llm::InferenceRequest {
            prompt: prompt.into(),
            parameters: &self.inference_params,
            play_back_previous_tokens: false,
            maximum_token_count: Some(120),
        },
        // OutputRequest
        &mut Default::default(),
        |r| match r {
            llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
                if t.contains(&self.stop_sequence) {
                    Ok(llm::InferenceFeedback::Halt)
                } else {
                    generated_text.push_str(&t); 
                    Ok(llm::InferenceFeedback::Continue)
                }
            }
            _ => Ok(llm::InferenceFeedback::Continue),
        },
    );


    generated_text 
}
}

pub fn call(prompt : String) -> String {
    let chatbot: Chatbot = Chatbot::new();
    let result: String = chatbot.process_user_input(&prompt);
    result
}

fn main() {
    call(prompt)
}