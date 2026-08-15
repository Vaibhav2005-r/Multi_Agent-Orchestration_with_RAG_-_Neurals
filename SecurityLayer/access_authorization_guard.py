import os
from typing import Dict
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class AccessAuthorizationGuard:
    """
    Uses an LLM to dynamically determine the required role for a given query,
    and checks if the provided user role meets or exceeds that requirement.
    Supported roles: EMPLOYEE, ADMIN.
    """
    
    # Role hierarchy: Higher number = Higher privilege
    ROLE_HIERARCHY: Dict[str, int] = {
        "EMPLOYEE": 1,
        "ADMIN": 2
    }
    
    def __init__(self, model_name: str = "meta/llama-3.1-8b-instruct"):
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY must be set in the environment variables.")
            
        self.llm = ChatNVIDIA(
            model=model_name,
            api_key=self.api_key,
            temperature=0.0 # Deterministic classification
        )
        
        # Build the chain to determine required role
        prompt = ChatPromptTemplate.from_template("""
        You are a Data Security Classifier. Your job is to determine the MINIMUM access role required to view information based on a user's query.
        The roles are hierarchical, from lowest to highest: EMPLOYEE, ADMIN.
        
        Guidelines:
        - EMPLOYEE: Standard operational inquiries, public regulations, compliance guidelines, RBI circulars, internal policies, standard procedures, documentation, and non-sensitive business data.
        - ADMIN: Highly sensitive data, private payroll details, executive compensation, mergers & acquisitions, PII, system configurations, credentials, and confidential financial audits.
        
        Based on the query below, respond with EXACTLY ONE WORD representing the minimum required role: EMPLOYEE or ADMIN.
        Do not add any other text.
        
        Query: {query}
        """)
        
        self.chain = prompt | self.llm | StrOutputParser()

    def determine_required_role(self, query: str) -> str:
        """Uses LLM to predict the required role for the query."""
        try:
            response = self.chain.invoke({"query": query}).strip().upper()
            # Clean up potential LLM verbosity just in case
            for role in self.ROLE_HIERARCHY.keys():
                if role in response:
                    return role
            # Fallback to highest security if confused
            return "ADMIN"
        except Exception as e:
            print(f"Error determining role via LLM: {e}")
            return "ADMIN" # Fail-secure approach

    def check_access(self, query: str, user_role: str = "EMPLOYEE") -> str:
        """
        Checks if the user_role is sufficient for the query.
        Returns the original query if allowed, or raises ValueError if denied.
        """
        user_role = user_role.upper()
        if user_role not in self.ROLE_HIERARCHY:
            print(f"Warning: Unknown user role '{user_role}'. Defaulting to EMPLOYEE.")
            user_role = "EMPLOYEE"
            
        required_role = self.determine_required_role(query)
        
        user_level = self.ROLE_HIERARCHY[user_role]
        required_level = self.ROLE_HIERARCHY[required_role]
        
        print(f"Auth Check - User Role: {user_role} (Lvl {user_level}), Required Role: {required_role} (Lvl {required_level})")
        
        if user_level >= required_level:
            return query
        else:
            raise ValueError(f"Query not allowed. Required role: {required_role}, your role: {user_role}.")

# =====================================================================
# Demo / Testing
# =====================================================================
if __name__ == "__main__":
    print("Initializing Access Authorization Guard...")
    auth_guard = AccessAuthorizationGuard()
    
    test_cases = [
        {"query": "What are the compliance rules for NBFCs?", "role": "EMPLOYEE"}, # Should pass
        {"query": "Show me the employee handbook for requesting PTO.", "role": "EMPLOYEE"}, # Should pass
        {"query": "What are the rules regarding loan disbursals and fees paid to LSPs?", "role": "EMPLOYEE"}, # Should pass
        {"query": "Show me the CEO's private payroll details.", "role": "EMPLOYEE"}, # Should fail
        {"query": "Show me the CEO's private payroll details.", "role": "ADMIN"}, # Should pass
    ]
    
    for case in test_cases:
        print(f"\n--- Query: '{case['query']}' | User: {case['role']} ---")
        try:
            auth_guard.check_access(case['query'], case['role'])
            print("Status: ACCESS GRANTED")
        except ValueError as ve:
            print(f"Status: ACCESS DENIED - {ve}")
