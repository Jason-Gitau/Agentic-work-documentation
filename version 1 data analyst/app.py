import streamlit as st
import pandas as pd
from memory_system import memory
from cost_tracker import cost_tracker
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from orchestrator import app as workflow

def main():
    st.set_page_config(
        page_title="Advanced Data Cleaning Agent",
        page_icon="🧹",
        layout="wide"
    )
    
    st.title("🧹 Advanced Data Cleaning Agent with LangGraph")
    st.markdown("*Smart, memory-driven data cleaning that learns from experience*")
    
    # Enhanced Sidebar Components
    with st.sidebar:
        st.header("📊 Performance Dashboard")
        
        # Memory Insights Section
        with st.expander("🧠 Memory Insights", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Patterns Learned", len(memory.successful_patterns))
            with col2:
                st.metric("Failed Executions", len(memory.failed_executions))
            
            if memory.successful_patterns:
                latest_pattern = max(
                    memory.successful_patterns.values(),
                    key=lambda x: x.last_success or datetime.min
                )
                st.caption(f"Last learned: {latest_pattern.last_success.strftime('%Y-%m-%d %H:%M')}")
            
            if memory.successful_patterns:
                pattern_usage = sorted(
                    [(fp, p.success_count) for fp, p in memory.successful_patterns.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.bar([f"Pattern {i+1}" for i in range(len(pattern_usage))],
                      [count for _, count in pattern_usage])
                ax.set_title("Most Used Patterns")
                ax.set_ylabel("Success Count")
                st.pyplot(fig)
        
        # Cost Tracking Section
        with st.expander("💰 Cost Tracking", expanded=True):
            total_cost = cost_tracker.get_total_cost()
            st.metric("Total Cost", f"${total_cost:.4f}")
            
            cost_by_type = {}
            for record in cost_tracker.records:
                if record.operation_type not in cost_by_type:
                    cost_by_type[record.operation_type] = 0.0
                cost_by_type[record.operation_type] += record.cost
            
            if cost_by_type:
                st.subheader("Cost Breakdown")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.pie(cost_by_type.values(), labels=cost_by_type.keys(),
                      autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
                
                st.subheader("Cost Over Time")
                cost_df = pd.DataFrame([{
                    'time': r.timestamp,
                    'cost': r.cost
                } for r in cost_tracker.records])
                
                if not cost_df.empty:
                    cost_df = cost_df.set_index('time').sort_index()
                    st.line_chart(cost_df['cost'].cumsum())
            
            st.subheader("Model Usage")
            model_costs = {}
            for record in cost_tracker.records:
                if record.model.value not in model_costs:
                    model_costs[record.model.value] = 0.0
                model_costs[record.model.value] += record.cost
            
            for model, cost in model_costs.items():
                st.metric(f"{model} Cost", f"${cost:.4f}")

    # Main Content
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Debug print
        # Start a new session for cost tracking
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cost_tracker.start_session(session_id)
        
        # Read data
        try:
            uploaded_file.seek(0)  # Reset pointer
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:", df.head())

        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
        
        if st.button("🚀 Start Cleaning", type="primary"):
            # Initialize state
            initial_state = {
                "file_path": uploaded_file,
                "current_data": df,
                "original_data": df.copy(),
                "schema_info": {},
                "quality_issues": [],
                "cleaning_plan": [],
                "current_step": 0,
                "generated_code": "",
                "execution_result": {},
                "is_complete": False
            }
            
            # Run workflow
            with st.spinner("Running data cleaning workflow..."):
                final_state = None
                for step in workflow.stream(initial_state):
                    node_name = list(step.keys())[0]
                    node_output = step[node_name]
                    
                    with st.expander(f"Step: {node_name.replace('_', ' ').title()}"):
                        if node_name == "read_context":
                            st.json(node_output.get("schema_info", {}))
                        elif node_name == "plan":
                            st.code(node_output.get("generated_code", ""), language="python")
                        elif node_name == "execute":
                            result = node_output.get("execution_result", {})
                            if result.get("success"):
                                st.success("✅ Execution successful")
                                st.dataframe(node_output.get("cleaned_data"))
                            else:
                                st.error(f"❌ Execution failed: {result.get('error')}")
                    
                    final_state = node_output
            
            if final_state and final_state.get("is_complete"):
                st.success("✅ Cleaning completed successfully!")
                st.write("Changes made:", final_state["original_data"].compare(final_state["current_data"]))
                st.download_button(
                    label="📥 Download Cleaned Data",
                    data=final_state["current_data"].to_csv(index=False),
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()