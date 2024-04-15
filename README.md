# Evaluation of Generative AI techniques for clinical report summarization

Our primary goal it to explore few of the prompt engineering techniques that will help assess the capabilities and limitations of LLMs for the healthcare domain specific summarization task. We evaluate the result with various promoting techniques. For more complex and clinical knowledge-intensive tasks, it's possible to build a language model-based system that accesses external knowledge sources to complete tasks. This enables more factual consistency, improves reliability of the generated responses, and helps to mitigate the problem of "hallucination", with RAG technique we demonstrate how the results are comparatively better. 

## Dataset 

The MIMIC Chest X-ray (MIMIC-CXR) Database v2.0.0 is a large publicly available dataset of chest radiographs in DICOM format with free-text radiology reports. We used the [MIMIC CXR dataset](https://physionet.org/content/mimic-cxr/2.0.0/) which can be accessed through a data use agreement, which requires user registration and completion of a credentialing process. During routine clinical care, clinicians trained in interpreting imaging studies (radiologists) will summarize their findings for a particular study in a free-text note. Radiology reports for the images were identified and extracted from the hospital EHR system. The reports were de-identified using a rule-based approach. Because we used only the radiology report text data, we downloaded just one compressed report file (mimic-cxr-reports.zip) from the MIMIC-CXR website. We used 2,000 reports (referred to as the dev1 dataset) from the separate held out subset of this dataset for evaluation. We use another 2,000 radiology reports (referred to as dev2) for evaluating from the chest X-ray collection from the Indiana University hospital network. 

## Techniques 

Here we explore 3 patterns :
1. Zero-Shot prompting 
2. Few-Shot prompting 
3. Retrieval augmented generation (RAG) 

Prompt Engineering is related to the Template Definition. Different templates can be used to express the same concept. Hence it is essential to carefully design the templates for exploiting the capability of a language model A prompt task is defined by a by prompt engineering, once the prompt template is defined, model generate multiple tokens that can fill a prompt template. For instance, “Generate radiology report impressions based on the following findings and output it within <impression> tags”. In this case, a model can fill the <impression> with tokens. 

### Zero-Shot prompting 
Zero-shot prompting implies providing a prompt to a large language model that is not part of the training data to the model. With a single prompt, the model should still generate a desired result. This technique makes large language models useful for many tasks and we used it to generate impressions from the findings section of a radiology report.
Here we have leverage Claude v2 model with Amazon Bedrock and provided prompt .

`prompt_zero_shot = """Human: Generate radiology report impressions based on the following findings and output it within <impression> tags. Findings: {}
Assistant:"""`

### Few-Shot prompting
Few-shot prompting leverages a small set of input-output examples to train the model for
specific tasks. The benefit of this technique is that it doesn’t require large amount of labelled
data (examples), and performs reasonably well by providing guidance to large language models.
In this work, we provided 5 examples of findings and impressions to the model for few-shot

examples_string = ''
for ex in examples:
    examples_string += f"""H:<findings>{ex['findings']}</findings>
    
A:<impression>{ex['impression']}</impression>\n"""
    
    
prompt_few_shot = """Human: Generate radiology report impressions based on the following findings. Findings: 
<findings>{}</findings>
        
Here are a few examples:
<examples>
""" + examples_string + """
</examples>

Assistant:"""

### Managed Retrieval augmented generation (RAG) with Amazon Bedrock Knowledge Base
To implement our RAG system, we utilized a dataset of 95,000 radiology report findings-impressions pairs as the knowledge source. This dataset was uploaded to S3 data source and then ingested using Amazon Bedrock Knowledge Base. We used Amazon Titan Text Embeddings model on Amazon Bedrock to generate vector embeddings. Embeddings are numerical representations of real-world objects that machine learning systems use to understand complex knowledge domains like humans do. The output vector representations were stored in a newly created vector store for efficient retrieval Amazon OpenSearch Serverless vector search collection. This leads to a public vector search collection and vector index set up with the required fields and necessary configurations. With the infrastructure in place, we set up a prompt template and leverage RetrieveandGenerate API for vector similarity search, and the Anthropic Claude 3 Sonnet model for impressions generation. Together, these components enabled both precise document retrieval and high-quality conditional text generation from the findings-to-impressions dataset.
The following reference architecture diagram in Figure 3 illustrates the fully managed RAG pattern with Amazon Bedrock Knowledge Base on AWS. The fully managed RAG provided by Knowledge Bases for Amazon Bedrock, converts user queries into embeddings, searches the knowledge base, obtains relevant results, augments the prompt and then invokes a LLM (Claude 3 Sonnet) to generate the response.

![Retrieval augmented generation pattern](images/Arch.png)
----------------------------------------------------------------------------------------------------------------------------------------
## Authors and acknowledgment
Special thanks to Wale Akinfaderin, Srushti Kotak , Ekta Walia Bhullar, Priya Padate and  for their contributions to this project and sharing their expertise in this field.

----------------------------------------------------------------------------------------------------------------------------------------
## License
This library is licensed under the MIT-0 License. See the LICENSE file.


