import pandas as pd
import plotly.graph_objects as go

# Create the data
data = [
  {"Step": 1, "Title": "Install Python", "Duration": "20 min", "Category": "Environment"},
  {"Step": 2, "Title": "Install VS Code", "Duration": "10 min", "Category": "Environment"},
  {"Step": 3, "Title": "Setup Python in VS Code", "Duration": "15 min", "Category": "Environment"},
  {"Step": 4, "Title": "Install Git", "Duration": "10 min", "Category": "Tools"},
  {"Step": 5, "Title": "Create Project Structure", "Duration": "15 min", "Category": "Project"},
  {"Step": 6, "Title": "Install Libraries", "Duration": "20 min", "Category": "Dependencies"},
  {"Step": 7, "Title": "Setup Database (SQLite)", "Duration": "10 min", "Category": "Database"},
  {"Step": 8, "Title": "Create Stock Data Fetcher", "Duration": "25 min", "Category": "Development"},
  {"Step": 9, "Title": "Create Dashboard", "Duration": "30 min", "Category": "Development"},
  {"Step": 10, "Title": "Environment Variables", "Duration": "10 min", "Category": "Configuration"},
  {"Step": 11, "Title": "Version Control Setup", "Duration": "10 min", "Category": "Tools"},
  {"Step": 12, "Title": "Test Everything", "Duration": "15 min", "Category": "Testing"},
  {"Step": 13, "Title": "Next Steps Planning", "Duration": "5 min", "Category": "Planning"}
]

df = pd.DataFrame(data)

# Define color mapping for categories using the provided brand colors
color_map = {
    'Environment': '#1FB8CD',    # Strong cyan
    'Tools': '#DB4545',          # Bright red  
    'Project': '#2E8B57',        # Sea green
    'Dependencies': '#5D878F',   # Cyan
    'Database': '#D2BA4C',       # Moderate yellow
    'Development': '#B4413C',    # Moderate red
    'Configuration': '#964325',   # Dark orange
    'Testing': '#944454',        # Pink-red
    'Planning': '#13343B'        # Dark cyan
}

# Abbreviate titles to fit character limits
df['Title_abbrev'] = df['Title'].apply(lambda x: x[:13] + '..' if len(x) > 15 else x)

# Create the flowchart
fig = go.Figure()

# Set positions for vertical flow
x_center = 5
y_positions = list(range(13, 0, -1))  # Step 1 at top (y=13), Step 13 at bottom (y=1)
box_width = 3
box_height = 0.7

# Add boxes and text for each step
for i, row in df.iterrows():
    y_pos = y_positions[i]
    color = color_map[row['Category']]
    
    # Add rectangle shape for box
    fig.add_shape(
        type="rect",
        x0=x_center - box_width/2,
        y0=y_pos - box_height/2,
        x1=x_center + box_width/2,
        y1=y_pos + box_height/2,
        fillcolor=color,
        line=dict(color="#333333", width=1),
        opacity=0.9
    )
    
    # Add step number and title text
    fig.add_annotation(
        x=x_center,
        y=y_pos + 0.15,
        text=f"<b>{row['Step']}. {row['Title_abbrev']}</b>",
        showarrow=False,
        font=dict(size=12, color="white"),
        xanchor="center",
        yanchor="middle"
    )
    
    # Add duration text
    fig.add_annotation(
        x=x_center,
        y=y_pos - 0.15,
        text=f"({row['Duration']})",
        showarrow=False,
        font=dict(size=10, color="white"),
        xanchor="center",
        yanchor="middle"
    )

# Add arrows between consecutive steps
for i in range(12):
    y_start = y_positions[i] - box_height/2 - 0.05
    y_end = y_positions[i+1] + box_height/2 + 0.05
    
    # Add arrow line
    fig.add_shape(
        type="line",
        x0=x_center,
        y0=y_start,
        x1=x_center,
        y1=y_end,
        line=dict(color="#333333", width=3)
    )
    
    # Add arrowhead
    fig.add_annotation(
        x=x_center,
        y=y_end,
        ax=x_center,
        ay=y_start,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor="#333333",
        showarrow=True,
        text=""
    )

# Update layout
fig.update_layout(
    title='Nifty 50 Setup Steps Flow',
    showlegend=False,
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 10]),
    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 14]),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Remove axis lines
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

# Update traces
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image('nifty50_setup_flowchart.png')