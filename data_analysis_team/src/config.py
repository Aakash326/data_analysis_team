# # config.py
# import os
# import litellm
# from litellm import completion

# # API Configuration
# GEMINI_API_KEY = "AIzaSyDloUkE9HfFg4ytufofLmxe3lmCVKZy8Dw"  # Replace with your actual API key

# # Set environment variables (CrewAI looks for these)
# os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
# os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY  # Some versions look for this

# # Configure LiteLLM globally
# litellm.api_key = GEMINI_API_KEY
# litellm.set_verbose = True  # Enable for debugging

# # LLM Configuration - Updated for better CrewAI compatibility
# def get_llm():
#     """Initialize and return the LLM instance compatible with CrewAI"""
    
#     class LiteLLMWrapper:
#         def __init__(self):
#             self.model = "gemini/gemini-2.0-flash"
#             self.temperature = 0.1
#             self.max_tokens = 4000
#             # Store API key as instance variable
#             self._api_key = GEMINI_API_KEY
        
#         def invoke(self, prompt, **kwargs):
#             """Main method for LLM invocation"""
#             try:
#                 # Handle different input formats more robustly
#                 if isinstance(prompt, str):
#                     messages = [{"role": "user", "content": prompt}]
#                 elif isinstance(prompt, list):
#                     # Check if it's already in message format
#                     if isinstance(prompt[0], dict) and "role" in prompt[0]:
#                         messages = prompt
#                     else:
#                         messages = [{"role": "user", "content": str(p)} for p in prompt]
#                 elif hasattr(prompt, 'content'):
#                     messages = [{"role": "user", "content": prompt.content}]
#                 elif hasattr(prompt, 'text'):
#                     messages = [{"role": "user", "content": prompt.text}]
#                 else:
#                     messages = [{"role": "user", "content": str(prompt)}]
                
#                 # Make the API call with explicit parameters
#                 response = completion(
#                     model=self.model,
#                     messages=messages,
#                     temperature=kwargs.get('temperature', self.temperature),
#                     max_tokens=kwargs.get('max_tokens', self.max_tokens),
#                     api_key=self._api_key
#                 )
                
#                 return response.choices[0].message.content
                
#             except Exception as e:
#                 print(f"LLM Error: {e}")
#                 # Try alternative model names if the first fails
#                 if "gemini/gemini-2.0-flash" in str(e):
#                     try:
#                         response = completion(
#                             model="gemini/gemini-pro",
#                             messages=messages,
#                             temperature=kwargs.get('temperature', self.temperature),
#                             max_tokens=kwargs.get('max_tokens', self.max_tokens),
#                             api_key=self._api_key
#                         )
#                         return response.choices[0].message.content
#                     except Exception as e2:
#                         print(f"Fallback LLM Error: {e2}")
                
#                 return f"Error in LLM response: {str(e)}"
        
#         def __call__(self, prompt, **kwargs):
#             """Alternative calling method"""
#             return self.invoke(prompt, **kwargs)
        
#         # Additional methods that CrewAI might expect
#         def predict(self, text, **kwargs):
#             return self.invoke(text, **kwargs)
        
#         def generate(self, prompts, **kwargs):
#             if isinstance(prompts, list):
#                 return [self.invoke(prompt, **kwargs) for prompt in prompts]
#             return self.invoke(prompts, **kwargs)
        
#         # CrewAI specific methods
#         def chat(self, messages, **kwargs):
#             return self.invoke(messages, **kwargs)
        
#         def complete(self, prompt, **kwargs):
#             return self.invoke(prompt, **kwargs)
        
#         # Properties that CrewAI might check
#         @property
#         def model_name(self):
#             return self.model
        
#         @property
#         def api_key(self):
#             return self._api_key
    
#     return LiteLLMWrapper()

# # Alternative simpler approach - Direct LiteLLM integration
# def get_simple_llm():
#     """Simpler LLM configuration that works with CrewAI's expectations"""
#     from langchain.llms.base import LLM
#     from typing import Optional, List, Any
    
#     class SimpleLiteLLM(LLM):
#         model: str = "gemini/gemini-2.0-flash"
#         temperature: float = 0.1
#         max_tokens: int = 4000
        
#         @property
#         def _llm_type(self) -> str:
#             return "litellm"
        
#         def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
#             try:
#                 response = completion(
#                     model=self.model,
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=self.temperature,
#                     max_tokens=self.max_tokens,
#                     api_key=GEMINI_API_KEY
#                 )
#                 return response.choices[0].message.content
#             except Exception as e:
#                 return f"Error: {str(e)}"
        
#         @property
#         def _identifying_params(self) -> dict:
#             return {
#                 "model": self.model,
#                 "temperature": self.temperature,
#                 "max_tokens": self.max_tokens
#             }
    
#     return SimpleLiteLLM()

# # Test function to verify API key works
# def test_gemini_connection():
#     """Test if Gemini API is working correctly"""
#     try:
#         response = completion(
#             model="gemini/gemini-1.5-flash",
#             messages=[{"role": "user", "content": "Hello, respond with 'API working' if you can read this."}],
#             api_key=GEMINI_API_KEY
#         )
#         print("✅ Gemini API test successful!")
#         print(f"Response: {response.choices[0].message.content}")
#         return True
#     except Exception as e:
#         print(f"❌ Gemini API test failed: {e}")
#         return False

# # Data Configuration
# DATA_CONFIG = {
#     "file_path": "/Users/saiaakash/Desktop/data.csv",  # Update path as needed
#     "target_column": "UnitPrice",
#     "date_column": "Date",
#     "categorical_columns": ["Description", "InvoiceDate", "Country"],
#     "numerical_columns": ["InvoiceNo", "StockCode", "Quantity", "UnitPrice", "CustomerID"]
# }

# # Analysis Configuration
# ANALYSIS_CONFIG = {
#     "outlier_methods": ["IQR", "Z-Score", "Isolation Forest"],
#     "correlation_threshold": 0.7,
#     "missing_value_threshold": 0.05,
#     "visualization_style": "seaborn",
#     "figure_size": (12, 8),
#     "random_state": 42
# }

# # Report Configuration
# REPORT_CONFIG = {
#     "output_format": "markdown",
#     "include_visualizations": True,
#     "max_insights": 10,
#     "significance_level": 0.05
# }

# # Run test when module is imported
# if __name__ == "__main__":
#     test_gemini_connection()
# config.py
import os
import litellm
from litellm import completion

# API Configuration
GEMINI_API_KEY = ""  # Replace with your actual API key

# Set environment variables (CrewAI looks for these)
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY  # Some versions look for this

# Configure LiteLLM globally
litellm.api_key = GEMINI_API_KEY
os.environ['LITELLM_LOG'] = 'DEBUG'  # Updated way to enable debugging

# LLM Configuration - Updated for better CrewAI compatibility
def get_llm():
    """Initialize and return the LLM instance compatible with CrewAI"""
    
    class LiteLLMWrapper:
        def __init__(self):
            self.model = "gemini/gemini-2.0-pro"
            self.temperature = 0.1
            self.max_tokens = 4000
            # Store API key as instance variable
            self._api_key = GEMINI_API_KEY
        
        def invoke(self, prompt, **kwargs):
            """Main method for LLM invocation"""
            try:
                # Handle different input formats more robustly
                if isinstance(prompt, str):
                    messages = [{"role": "user", "content": prompt}]
                elif isinstance(prompt, list):
                    # Check if it's already in message format
                    if isinstance(prompt[0], dict) and "role" in prompt[0]:
                        messages = prompt
                    else:
                        messages = [{"role": "user", "content": str(p)} for p in prompt]
                elif hasattr(prompt, 'content'):
                    messages = [{"role": "user", "content": prompt.content}]
                elif hasattr(prompt, 'text'):
                    messages = [{"role": "user", "content": prompt.text}]
                else:
                    messages = [{"role": "user", "content": str(prompt)}]
                
                print(f"DEBUG: Sending messages: {messages}")  # Debug log
                
                # Make the API call with explicit parameters
                response = completion(
                    model=self.model,
                    messages=messages,
                    temperature=kwargs.get('temperature', self.temperature),
                    max_tokens=kwargs.get('max_tokens', self.max_tokens),
                    api_key=self._api_key
                )
                
                print(f"DEBUG: Raw response: {response}")  # Debug log
                
                # Check if response has the expected structure
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    if hasattr(response.choices[0], 'message'):
                        content = response.choices[0].message.content
                    elif hasattr(response.choices[0], 'text'):
                        content = response.choices[0].text
                    else:
                        content = str(response.choices[0])
                else:
                    # Fallback for different response formats
                    content = str(response)
                
                print(f"DEBUG: Extracted content: {content}")  # Debug log
                return content
                
            except Exception as e:
                print(f"LLM Error: {e}")
                print(f"Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                
                # Try alternative model names if the first fails
                if "gemini/gemini-2.0-pro" in str(e):
                    try:
                        print("Trying fallback model: gemini/gemini-pro")
                        response = completion(
                            model="gemini/gemini-pro",
                            messages=messages,
                            temperature=kwargs.get('temperature', self.temperature),
                            max_tokens=kwargs.get('max_tokens', self.max_tokens),
                            api_key=self._api_key
                        )
                        
                        if hasattr(response, 'choices') and len(response.choices) > 0:
                            if hasattr(response.choices[0], 'message'):
                                return response.choices[0].message.content
                            elif hasattr(response.choices[0], 'text'):
                                return response.choices[0].text
                            else:
                                return str(response.choices[0])
                        
                    except Exception as e2:
                        print(f"Fallback LLM Error: {e2}")
                
                return f"Error in LLM response: {str(e)}"
        
        def __call__(self, prompt, **kwargs):
            """Alternative calling method"""
            return self.invoke(prompt, **kwargs)
        
        # Additional methods that CrewAI might expect
        def predict(self, text, **kwargs):
            return self.invoke(text, **kwargs)
        
        def generate(self, prompts, **kwargs):
            if isinstance(prompts, list):
                return [self.invoke(prompt, **kwargs) for prompt in prompts]
            return self.invoke(prompts, **kwargs)
        
        # CrewAI specific methods
        def chat(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)
        
        def complete(self, prompt, **kwargs):
            return self.invoke(prompt, **kwargs)
        
        # Properties that CrewAI might check
        @property
        def model_name(self):
            return self.model
        
        @property
        def api_key(self):
            return self._api_key
    
    return LiteLLMWrapper()

# Alternative simpler approach - Direct LiteLLM integration
def get_simple_llm():
    """Simpler LLM configuration that works with CrewAI's expectations"""
    from langchain.llms.base import LLM
    from typing import Optional, List, Any
    
    class SimpleLiteLLM(LLM):
        model: str = "gemini/gemini-2.0-pro"
        temperature: float = 0.1
        max_tokens: int = 4000
        
        @property
        def _llm_type(self) -> str:
            return "litellm"
        
        def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
            try:
                response = completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=GEMINI_API_KEY
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error: {str(e)}"
        
        @property
        def _identifying_params(self) -> dict:
            return {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
    
    return SimpleLiteLLM()

# Test function to verify API key works
def test_gemini_connection():
    """Test if Gemini API is working correctly"""
    try:
        response = completion(
            model="gemini/gemini-2.0-pro",
            messages=[{"role": "user", "content": "Hello, respond with 'API working' if you can read this."}],
            api_key=GEMINI_API_KEY
        )
        print("✅ Gemini API test successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False

# Data Configuration
DATA_CONFIG = {
    "file_path": "/Users/saiaakash/Desktop/data.csv",  # Update path as needed
    "target_column": "UnitPrice",
    "date_column": "Date",
    "categorical_columns": ["Description", "InvoiceDate", "Country"],
    "numerical_columns": ["InvoiceNo", "StockCode", "Quantity", "UnitPrice", "CustomerID"]
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    "outlier_methods": ["IQR", "Z-Score", "Isolation Forest"],
    "correlation_threshold": 0.7,
    "missing_value_threshold": 0.05,
    "visualization_style": "seaborn",
    "figure_size": (12, 8),
    "random_state": 42
}

# Report Configuration
REPORT_CONFIG = {
    "output_format": "markdown",
    "include_visualizations": True,
    "max_insights": 10,
    "significance_level": 0.05
}

# Run test when module is imported
if __name__ == "__main__":
    test_gemini_connection()