import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import seaborn as sns

import plotly.graph_objects as go

# Race 
race_code = {"White":1, "Black":2, "Hispanic":3, "Asian":4, "Native American":5, "Pacific Islander":6, "Other":8}

class BodySkeleton:
    def __init__(self, df, height, gender="Both", race="All", ratio=False):
        # stature df for gender and race 
        if race != "All":
            dff = df[(df['DODRace'] == race_code[race])]
        else:
            dff = df.copy()
        
        if gender != "Both":
            dff = dff[(dff["Gender"] == gender)]
         
        # If height is not all, get subset of the dataframe
        if isinstance(height, str) and height != "All":
            h1 = int(height)*10 - 5
            h2 = int(height)*10 + 5
            
            dff = dff[(dff["stature"] >= h1) & (dff["stature"] <= h2)]
            self.height = int(height)
        else:
            self.height = int(dff["stature"].mean()/10)
            
        self.ratio = ratio
        self.df = dff
        self.gender = gender
        self.race = race
        self.arm_angle = np.pi/3 # 60 degrees
        
        self.fig = go.Figure()
        self.line_color = "black"
        self.label_padding = 20

        self.x_ref = {
            "stature": 10,
            "body_height": 20,
            "leg_height": 30,
            "left_boundary": 100
        }
        
        self.y_ref = {
            "floor": 5
        }
        
    def drawBody(self):
        # using the stature or height dimension calculate the proportion and dimension
        # and also the center_body location as a reference point
        dff = self.df
        
        # Body 
        body_p = dff["acromialheight_pconstant"].mean()
        body_height = body_p*self.height
        
        # Shoulder
        shoulder_p = (((dff["biacromialbreadth"] + dff["bideltoidbreadth"])/2)/dff["stature"]).mean()
        shoulder_width =  shoulder_p*self.height
        
        # left shoulder
        lshoulder_joint = (self.x_ref["left_boundary"], body_height+self.y_ref["floor"])
        # right shoulder
        rshoulder_joint = (self.x_ref["left_boundary"]+shoulder_width, body_height+self.y_ref["floor"])
        
        # Draw Shoulder
        self.drawLine(lshoulder_joint, rshoulder_joint, "Shoulder", shoulder_width)
        
        # Mid-Point or Body
        body_x = self.x_ref["left_boundary"] + (shoulder_width/2)
        
        # Torso
        torso_p = ((dff["acromialheight"] - dff["trochanterionheight"])/dff["stature"]).mean()
        torso_length = torso_p*self.height
        
        # Lower Body Length
        lower_p = dff["trochanterionheight_pconstant"].mean()
        lower_length = lower_p*self.height
        
        # Shoulder Midpoint
        midpoint_shoulder = (body_x, body_height+self.y_ref["floor"])
        
        # Pelvic Midpoint 
        midpoint_pelvis = (body_x, lower_length+self.y_ref["floor"])
        
        # Draw Torso
        self.drawLine(midpoint_shoulder, midpoint_pelvis, "Torso", torso_length)
        
        # Hip
        hip_p = dff["hipbreadth_pconstant"].mean()
        hip_width = hip_p*self.height
        
        # Left Pelvic Point
        lpelvic_joint = (body_x-(hip_width/2), lower_length+self.y_ref["floor"])
        # Right Pelvic Point
        rpelvic_joint = (body_x+(hip_width/2), lower_length+self.y_ref["floor"])
        
        # Draw Hip
        self.drawLine(lpelvic_joint, rpelvic_joint, "Hip", hip_width)
        
        # Humerus 
        humerus_p = ((dff["sleeveoutseam"] - dff["radialestylionlength"])/dff["stature"]).mean()
        humerus_length = humerus_p*self.height
        
        # Left Elbow Joint
        lelbow_joint = (self.x_ref["left_boundary"], body_height-humerus_length)
        
        x_ratio = np.cos(self.arm_angle)
        y_ratio = np.sin(self.arm_angle)
        # Right Elbow Joint
        relbow_joint = (self.x_ref["left_boundary"]+shoulder_width+(x_ratio*humerus_length), \
                        body_height-(y_ratio*humerus_length))
        
        # Draw left Humerus
        self.drawLine(lshoulder_joint, lelbow_joint, "Left Humerus", None)
        
        # Draw Right Humerus
        self.drawLine(rshoulder_joint, relbow_joint, "Right Humerus", humerus_length)
        
        # Radius
        radius_p = dff["radialestylionlength_pconstant"].mean()
        radius_length = radius_p*self.height
        
        # Left Wrist Joint
        lwrist_joint = (self.x_ref["left_boundary"], body_height-humerus_length-radius_length)
        # Right Wrist Joint
        rwrist_joint = (self.x_ref["left_boundary"]+shoulder_width+(x_ratio*humerus_length)+(x_ratio*radius_length), \
                        body_height-(humerus_length+radius_length)*y_ratio)
        
        # Draw Left Radius
        self.drawLine(lelbow_joint, lwrist_joint, "Left Radius", None)
        
        # Draw Right Radius
        self.drawLine(relbow_joint, rwrist_joint, "Right Radius", radius_length)
        
        # Hand 
        hand_p = dff["handlength_pconstant"].mean()
        hand_length = hand_p*self.height
        
        # Left Hand
        lhand = (self.x_ref["left_boundary"], body_height-humerus_length-radius_length-hand_length)
        # Right Hand
        rhand = (self.x_ref["left_boundary"]+shoulder_width+x_ratio*(humerus_length+radius_length+hand_length), \
                 body_height-(humerus_length+radius_length+hand_length)*y_ratio)
        
        # Draw Left Hand
        self.drawLine(lwrist_joint, lhand, "Left Hand", None)
        
        # Draw Right Hand
        self.drawLine(rwrist_joint, rhand, "Right Hand", hand_length)
        
        # Femur
        femur_p = ((dff["trochanterionheight"] - dff["lateralfemoralepicondyleheight"])/dff["stature"]).mean()
        femur_length = femur_p*self.height
        
        # Tibia
        tibia_p = ((dff["lateralfemoralepicondyleheight"] - dff["lateralmalleolusheight"])/ dff["stature"]).mean()
        tibia_length = tibia_p*self.height
        
        # Ankle 
        ankle_p = dff["lateralmalleolusheight_pconstant"].mean()
        ankle_length = ankle_p*self.height
        
        # Left Knee
        lknee_joint = (lpelvic_joint[0], tibia_length+ankle_length+self.y_ref["floor"])
        # Right Knee
        rknee_joint = (rpelvic_joint[0], tibia_length+ankle_length+self.y_ref["floor"])
        
        # Draw Left Femur
        self.drawLine(lpelvic_joint, lknee_joint, "Left Femur", None)
        
        # Draw Right Femur
        self.drawLine(rpelvic_joint, rknee_joint, "Right Femur", femur_length)
        
        # Left Ankle
        lankle_joint = (lpelvic_joint[0], ankle_length+self.y_ref["floor"])
        # Right Ankle
        rankle_joint = (rpelvic_joint[0], ankle_length+self.y_ref["floor"])
        
        # Draw Left Tibia
        self.drawLine(lknee_joint, lankle_joint, "Left Tibia", None)
        
        # Draw Right Tibia
        self.drawLine(rknee_joint, rankle_joint, "Right Tibia", tibia_length)
        
        # Left Foot
        lfoot = (lpelvic_joint[0], self.y_ref["floor"])
        # Right Foot
        rfoot = (rpelvic_joint[0], self.y_ref["floor"])
        
        # Draw Left Foot
        self.drawLine(lankle_joint, lfoot, "Left Foot", None)
        
        # Draw Right Foot
        self.drawLine(rankle_joint, rfoot, "Right Foot", ankle_length)
        
        # Draw Neck and Head
        neck_head_height = self.height - body_height
        head_height = (3/4)*neck_head_height
        neck_height = (1/4)*neck_head_height
        radius = head_height/2
        center = (body_x, body_height + neck_height + radius + self.y_ref["floor"])
        
        self.fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=center[0]-radius, y0=center[1]-radius, x1=center[0]+radius, y1=center[1]+radius,
            line_color="Black"
        )
        
        # Draw Neck
        self.drawLine((body_x, body_height+self.y_ref["floor"]), (body_x, neck_height+body_height+self.y_ref["floor"]), "Neck", \
                      neck_height)
        
    def drawLine(self, p1, p2, name, label=None):
        # draw points
        x = [p1[0], p2[0]]
        y = [p1[1], p2[1]]
        
        # as an example for where to place the text we can use the mean
        xmean = sum(i for i in x) / float(len(x))
        ymean = sum(i for i in y) / float(len(y))
        
        self.fig.add_trace(go.Scatter(
            x=x,
            y=y
        ))
            
        # Create scatter trace of text labels
        if label is not None and is_float(label):
            self.fig.add_trace(go.Scatter(
                x=[xmean+self.label_padding],
                y=[ymean],
                text=[round(float(label), 3) if self.ratio == "Absolute" else round(float(label)/self.height, 3)],
                name=name,
                mode="text",
            ))
        elif label is not None:
            self.fig.add_trace(go.Scatter(
                x=[xmean+self.label_padding],
                y=[ymean],
                text=[label],
                name=name,
                mode="text",
            ))

        # Draw Line
        self.fig.add_shape(type="line",
            x0=p1[0], y0=p1[1], x1=p2[0], y1=p2[1],
            line=dict(color=self.line_color, width=3)
        )
        
    def getFig(self):
        
        if len(self.df) != 0:
            # draw height or stature
            self.drawLine((self.x_ref["stature"], self.y_ref["floor"]), (self.x_ref["stature"], self.y_ref["floor"]+self.height), \
                          "Stature", str(self.height))

            # body Height
            body_p = self.df["acromialheight_pconstant"].mean()

            body_height = round(body_p*self.height, 2)

            # draw body height
            self.drawLine((self.x_ref["body_height"], self.y_ref["floor"]), (self.x_ref["body_height"], self.y_ref["floor"]+body_height), \
                          "Body Height", str(body_height))

            # Total Leg Length
            leg_p = self.df["trochanterionheight_pconstant"].mean()
            leg_height = leg_p*self.height

            self.drawLine((self.x_ref["leg_height"], self.y_ref["floor"]), (self.x_ref["leg_height"], self.y_ref["floor"]+leg_height), \
                          "Leg Height", str(leg_height))

            self.drawBody()
        
        # Graph Size and Range
        self.fig.update_layout(yaxis_range=[0, 250])
        self.fig.update_layout(xaxis_range=[0, 250])
        self.fig.update_layout(width=int(500))
        self.fig.update_layout(height=int(500))
        
        # Axis Titles
        self.fig.update_xaxes(title_text='Width (cm)')
        self.fig.update_yaxes(title_text='Height (cm)')
        
        # Remove Trace Names
        # set showlegend property by name of trace
        for trace in self.fig['data']: 
            if(trace['name'] == None): trace['showlegend'] = False

        return self.fig
        
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False