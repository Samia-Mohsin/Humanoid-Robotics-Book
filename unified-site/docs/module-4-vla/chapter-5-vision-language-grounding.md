# Chapter 5: Vision-Language Grounding

## Introduction to Vision-Language Grounding

Vision-Language Grounding is a critical capability for humanoid robots, enabling them to connect linguistic descriptions with visual perceptions in the real world. This capability allows robots to understand commands like "pick up the red cup on the left" by identifying the specific object in their visual field that corresponds to the linguistic description.

Vision-language grounding involves:
- Associating words and phrases with visual entities in the environment
- Understanding spatial relationships described in natural language
- Connecting abstract concepts to concrete visual observations
- Maintaining consistent mappings between language and vision across different contexts

## Object Grounding

Object grounding is the process of linking linguistic references to specific visual objects in the environment. This is fundamental for humanoid robots to execute commands that involve manipulating specific objects.

### Visual Feature Extraction

Effective object grounding requires extracting meaningful features from visual input that can be associated with linguistic descriptions.

```python
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import cv2
from transformers import CLIPModel, CLIPProcessor

@dataclass
class VisualObject:
    id: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    features: np.ndarray
    attributes: Dict[str, Any]
    confidence: float

class VisualFeatureExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.object_detector = self.initialize_object_detector()

    def initialize_object_detector(self):
        """
        Initialize object detection model for extracting objects from images
        """
        # This would typically use a model like YOLO, DETR, or similar
        return {
            'model': 'object_detection_model',
            'confidence_threshold': 0.7,
            'classes': ['person', 'chair', 'table', 'cup', 'bottle', 'book', 'phone', 'keys']
        }

    def extract_objects(self, image: np.ndarray) -> List[VisualObject]:
        """
        Extract objects from an image with their visual features
        """
        # Detect objects in the image
        detections = self.detect_objects(image)

        objects = []
        for detection in detections:
            # Extract visual features for the detected object
            bbox = detection['bbox']
            object_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            # Extract CLIP features for the object
            features = self.extract_clip_features(object_image)

            # Extract additional attributes
            attributes = self.extract_attributes(object_image, detection)

            visual_object = VisualObject(
                id=detection['id'],
                bbox=bbox,
                features=features,
                attributes=attributes,
                confidence=detection['confidence']
            )

            objects.append(visual_object)

        return objects

    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in the image using object detection model
        """
        # For demonstration, returning mock detections
        # In reality, this would use an actual object detection model
        height, width = image.shape[:2]

        mock_detections = [
            {
                'id': 'obj1',
                'bbox': [100, 100, 200, 200],  # x1, y1, x2, y2
                'class': 'cup',
                'confidence': 0.92,
                'center': (150, 150)
            },
            {
                'id': 'obj2',
                'bbox': [250, 150, 350, 250],
                'class': 'book',
                'confidence': 0.88,
                'center': (300, 200)
            },
            {
                'id': 'obj3',
                'bbox': [50, 200, 150, 300],
                'class': 'phone',
                'confidence': 0.95,
                'center': (100, 250)
            }
        ]

        return mock_detections

    def extract_clip_features(self, object_image: np.ndarray) -> np.ndarray:
        """
        Extract CLIP features for an object image
        """
        inputs = self.processor(images=object_image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)

        return outputs.cpu().numpy()

    def extract_attributes(self, object_image: np.ndarray, detection: Dict) -> Dict[str, Any]:
        """
        Extract additional attributes like color, size, shape from object image
        """
        # Basic attribute extraction
        height, width = object_image.shape[:2]

        # Color analysis
        avg_color = np.mean(object_image, axis=(0, 1))
        color_names = ['red', 'green', 'blue']
        dominant_color_idx = np.argmax(avg_color)
        dominant_color = color_names[dominant_color_idx]

        # Size classification
        area = height * width
        if area < 5000:
            size = 'small'
        elif area < 15000:
            size = 'medium'
        else:
            size = 'large'

        # Shape approximation (simplified)
        aspect_ratio = width / height
        if 0.8 <= aspect_ratio <= 1.2:
            shape = 'square'
        elif aspect_ratio > 1.5:
            shape = 'horizontal'
        else:
            shape = 'vertical'

        return {
            'color': dominant_color,
            'size': size,
            'shape': shape,
            'class': detection['class'],
            'area': area,
            'aspect_ratio': aspect_ratio
        }
```

### Language-Object Association

The core of object grounding is associating linguistic descriptions with visual objects based on their features and attributes.

```python
class LanguageObjectAssociator:
    def __init__(self):
        self.similarity_threshold = 0.7
        self.feature_weights = {
            'color': 0.3,
            'shape': 0.2,
            'size': 0.2,
            'class': 0.3
        }

    def ground_objects(self, objects: List[VisualObject],
                      linguistic_description: str) -> List[Tuple[VisualObject, float]]:
        """
        Ground linguistic description to visual objects
        """
        # Parse the linguistic description
        description_features = self.parse_description(linguistic_description)

        # Calculate similarity between description and each object
        similarities = []
        for obj in objects:
            similarity = self.calculate_similarity(description_features, obj.attributes)
            similarities.append((obj, similarity))

        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities

    def parse_description(self, description: str) -> Dict[str, Any]:
        """
        Parse linguistic description to extract relevant features
        """
        description_lower = description.lower()

        # Extract color keywords
        color_keywords = {
            'red': ['red', 'crimson', 'scarlet', 'ruby'],
            'blue': ['blue', 'azure', 'navy', 'sapphire'],
            'green': ['green', 'emerald', 'forest', 'olive'],
            'yellow': ['yellow', 'gold', 'amber', 'lemon'],
            'black': ['black', 'dark', 'ebony'],
            'white': ['white', 'light', 'ivory'],
            'brown': ['brown', 'tan', 'chocolate', 'copper']
        }

        # Extract size keywords
        size_keywords = {
            'small': ['small', 'tiny', 'little', 'mini', 'compact'],
            'medium': ['medium', 'average', 'normal', 'regular'],
            'large': ['large', 'big', 'huge', 'massive', 'giant']
        }

        # Extract shape keywords
        shape_keywords = {
            'square': ['square', 'rectangular', 'box', 'cube'],
            'round': ['round', 'circular', 'spherical', 'cylinder'],
            'tall': ['tall', 'vertical', 'elongated'],
            'wide': ['wide', 'horizontal', 'broad']
        }

        # Identify object classes
        class_keywords = ['cup', 'bottle', 'book', 'phone', 'keys', 'table', 'chair', 'person']

        parsed_features = {
            'color': None,
            'size': None,
            'shape': None,
            'class': None,
            'spatial_relations': []
        }

        # Identify color
        for color, keywords in color_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                parsed_features['color'] = color
                break

        # Identify size
        for size, keywords in size_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                parsed_features['size'] = size
                break

        # Identify shape
        for shape, keywords in shape_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                parsed_features['shape'] = shape
                break

        # Identify class
        for class_name in class_keywords:
            if class_name in description_lower:
                parsed_features['class'] = class_name
                break

        # Extract spatial relations
        spatial_keywords = ['left', 'right', 'front', 'back', 'behind', 'in front of',
                           'next to', 'near', 'on', 'under', 'above', 'below']
        for keyword in spatial_keywords:
            if keyword in description_lower:
                parsed_features['spatial_relations'].append(keyword)

        return parsed_features

    def calculate_similarity(self, description_features: Dict,
                           object_attributes: Dict) -> float:
        """
        Calculate similarity between description and object
        """
        total_score = 0.0
        total_weight = 0.0

        # Color similarity
        if description_features['color'] and object_attributes['color']:
            color_score = 1.0 if description_features['color'] == object_attributes['color'] else 0.0
            total_score += self.feature_weights['color'] * color_score
            total_weight += self.feature_weights['color']

        # Size similarity
        if description_features['size'] and object_attributes['size']:
            size_score = 1.0 if description_features['size'] == object_attributes['size'] else 0.0
            total_score += self.feature_weights['size'] * size_score
            total_weight += self.feature_weights['size']

        # Class similarity
        if description_features['class'] and object_attributes['class']:
            class_score = 1.0 if description_features['class'] == object_attributes['class'] else 0.0
            total_score += self.feature_weights['class'] * class_score
            total_weight += self.feature_weights['class']

        # Shape similarity
        if description_features['shape'] and object_attributes['shape']:
            shape_score = 1.0 if description_features['shape'] == object_attributes['shape'] else 0.0
            total_score += self.feature_weights['shape'] * shape_score
            total_weight += self.feature_weights['shape']

        # If no features matched, return 0
        if total_weight == 0:
            return 0.0

        return total_score / total_weight
```

## Spatial Grounding

Spatial grounding enables humanoid robots to understand spatial relationships described in natural language and map them to visual spatial configurations in the environment.

### Spatial Relationship Detection

Detecting and understanding spatial relationships between objects is crucial for tasks involving navigation and manipulation.

```python
class SpatialRelationshipDetector:
    def __init__(self):
        self.spatial_relations = [
            'left_of', 'right_of', 'in_front_of', 'behind',
            'above', 'below', 'on_top_of', 'under', 'next_to',
            'near', 'far_from', 'between', 'inside', 'outside'
        ]

        # Define spatial relation computation parameters
        self.angle_threshold = 20  # degrees for directional relations
        self.distance_threshold = 1.0  # meters for near/far

    def detect_spatial_relationships(self, objects: List[VisualObject]) -> List[Dict]:
        """
        Detect spatial relationships between objects
        """
        relationships = []

        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    # Calculate spatial relationship
                    relation = self.compute_spatial_relation(obj1, obj2)
                    if relation:
                        relationship = {
                            'subject': obj1.id,
                            'relation': relation,
                            'object': obj2.id,
                            'confidence': self.calculate_relation_confidence(obj1, obj2, relation)
                        }
                        relationships.append(relationship)

        return relationships

    def compute_spatial_relation(self, obj1: VisualObject, obj2: VisualObject) -> str:
        """
        Compute spatial relationship between two objects
        """
        # Get 3D positions of objects (simplified using 2D center points)
        # In reality, this would use depth information
        pos1 = self.get_object_position(obj1)
        pos2 = self.get_object_position(obj2)

        dx = pos2[0] - pos1[0]  # x difference
        dy = pos2[1] - pos1[1]  # y difference (depth)

        # Calculate angle and distance
        angle = np.arctan2(dy, dx) * 180 / np.pi  # angle in degrees
        distance = np.sqrt(dx**2 + dy**2)

        # Determine spatial relation based on angle and distance
        if abs(dx) > abs(dy):  # More horizontal than vertical
            if dx > 0:
                return 'right_of'
            else:
                return 'left_of'
        else:  # More vertical than horizontal
            if dy > 0:
                return 'in_front_of'
            else:
                return 'behind'

    def get_object_position(self, obj: VisualObject) -> Tuple[float, float]:
        """
        Get approximate 2D position of object (simplified)
        """
        # Extract center of bounding box
        x1, y1, x2, y2 = obj.bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Normalize to relative position in image
        # In reality, this would be converted to 3D world coordinates
        return (center_x, center_y)

    def calculate_relation_confidence(self, obj1: VisualObject,
                                   obj2: VisualObject,
                                   relation: str) -> float:
        """
        Calculate confidence for a spatial relationship
        """
        # Higher confidence if objects are clearly separated
        x1, y1, x2, y2 = obj1.bbox
        x3, y3, x4, y4 = obj2.bbox

        # Calculate overlap
        overlap_area = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
        obj1_area = (x2 - x1) * (y2 - y1)

        # Lower confidence if objects overlap significantly
        overlap_ratio = overlap_area / obj1_area if obj1_area > 0 else 0
        confidence = max(0.5, 1.0 - overlap_ratio)  # At least 0.5 confidence

        return confidence
```

### Spatial Language Understanding

Understanding spatial language requires connecting linguistic spatial expressions with visual spatial configurations.

```python
class SpatialLanguageProcessor:
    def __init__(self):
        self.spatial_prepositions = {
            'left': ['left', 'left side', 'to the left', 'on the left'],
            'right': ['right', 'right side', 'to the right', 'on the right'],
            'front': ['front', 'in front', 'in front of', 'ahead of'],
            'back': ['back', 'behind', 'in back of', 'at the back'],
            'above': ['above', 'over', 'on top of', 'up'],
            'below': ['below', 'under', 'underneath', 'beneath', 'down'],
            'near': ['near', 'close to', 'next to', 'beside', 'by'],
            'far': ['far', 'away from', 'distant from']
        }

    def parse_spatial_description(self, description: str) -> Dict[str, Any]:
        """
        Parse spatial description from natural language
        """
        description_lower = description.lower()

        parsed_spatial = {
            'reference_object': None,
            'spatial_relation': None,
            'target_object': None,
            'spatial_preposition': None
        }

        # Extract spatial prepositions
        for relation, prepositions in self.spatial_prepositions.items():
            for prep in prepositions:
                if prep in description_lower:
                    parsed_spatial['spatial_relation'] = relation
                    parsed_spatial['spatial_preposition'] = prep
                    break
            if parsed_spatial['spatial_relation']:
                break

        # Extract object references (simplified)
        # This would typically use more sophisticated NLP
        words = description_lower.split()
        object_keywords = ['cup', 'bottle', 'book', 'phone', 'table', 'chair', 'person']

        # Find target and reference objects
        if parsed_spatial['spatial_preposition']:
            prep_index = description_lower.find(parsed_spatial['spatial_preposition'])
            before_prep = description_lower[:prep_index].strip()
            after_prep = description_lower[prep_index + len(parsed_spatial['spatial_preposition']):].strip()

            # Find objects in both parts
            for obj in object_keywords:
                if obj in before_prep:
                    parsed_spatial['target_object'] = obj
                if obj in after_prep:
                    parsed_spatial['reference_object'] = obj

        return parsed_spatial

    def ground_spatial_description(self, description: str,
                                 objects: List[VisualObject],
                                 relationships: List[Dict]) -> List[VisualObject]:
        """
        Ground spatial description to specific objects
        """
        parsed_spatial = self.parse_spatial_description(description)

        # Find reference object
        reference_obj = None
        for obj in objects:
            if parsed_spatial['reference_object'] and parsed_spatial['reference_object'] in obj.attributes.get('class', ''):
                reference_obj = obj
                break

        if not reference_obj:
            # If no reference object found, return all objects matching target description
            target_objects = []
            for obj in objects:
                if parsed_spatial['target_object'] and parsed_spatial['target_object'] in obj.attributes.get('class', ''):
                    target_objects.append(obj)
            return target_objects

        # Find objects that satisfy the spatial relationship
        matching_objects = []
        for rel in relationships:
            if rel['object'] == reference_obj.id and rel['relation'] == parsed_spatial['spatial_relation']:
                # Find the subject object
                for obj in objects:
                    if obj.id == rel['subject']:
                        matching_objects.append(obj)

        return matching_objects
```

## Grounding in Context

Effective vision-language grounding must consider the broader context including scene context, task context, and social context.

### Context-Aware Grounding

Context-aware grounding improves accuracy by considering additional information beyond just visual and linguistic features.

```python
class ContextAwareGrounding:
    def __init__(self):
        self.scene_context = {}
        self.task_context = {}
        self.social_context = {}

    def update_context(self, scene_description: Dict,
                      task_description: Dict,
                      social_description: Dict):
        """
        Update contextual information for grounding
        """
        self.scene_context = scene_description
        self.task_context = task_description
        self.social_context = social_description

    def ground_with_context(self, linguistic_description: str,
                          visual_objects: List[VisualObject],
                          spatial_relationships: List[Dict]) -> List[Tuple[VisualObject, float]]:
        """
        Ground linguistic description with contextual information
        """
        # Perform basic grounding
        basic_groundings = self.basic_ground(linguistic_description, visual_objects)

        # Apply contextual refinements
        contextual_groundings = []
        for obj, base_score in basic_groundings:
            context_score = self.calculate_context_score(
                obj, linguistic_description, spatial_relationships
            )

            # Combine base score with context score
            final_score = self.combine_scores(base_score, context_score)
            contextual_groundings.append((obj, final_score))

        # Sort by final score
        contextual_groundings.sort(key=lambda x: x[1], reverse=True)

        return contextual_groundings

    def basic_ground(self, description: str, objects: List[VisualObject]) -> List[Tuple[VisualObject, float]]:
        """
        Perform basic grounding without context
        """
        associator = LanguageObjectAssociator()
        return associator.ground_objects(objects, description)

    def calculate_context_score(self, obj: VisualObject,
                              description: str,
                              relationships: List[Dict]) -> float:
        """
        Calculate context-based score for an object
        """
        context_score = 0.0
        total_weight = 0.0

        # Scene context: is the object in an expected location?
        scene_context_weight = 0.3
        if self.scene_context.get('room_type') == 'kitchen':
            if obj.attributes.get('class') in ['cup', 'plate', 'bottle']:
                context_score += scene_context_weight * 0.8
            elif obj.attributes.get('class') in ['sofa', 'tv']:
                context_score += scene_context_weight * 0.1
        elif self.scene_context.get('room_type') == 'living_room':
            if obj.attributes.get('class') in ['sofa', 'tv', 'coffee_table']:
                context_score += scene_context_weight * 0.8
            elif obj.attributes.get('class') in ['oven', 'fridge']:
                context_score += scene_context_weight * 0.1

        total_weight += scene_context_weight

        # Task context: is the object relevant to the current task?
        task_context_weight = 0.4
        if self.task_context.get('task_type') == 'serving_drinks':
            if obj.attributes.get('class') in ['cup', 'bottle', 'glass']:
                context_score += task_context_weight * 0.9
            elif obj.attributes.get('class') in ['book', 'phone']:
                context_score += task_context_weight * 0.2
        elif self.task_context.get('task_type') == 'reading':
            if obj.attributes.get('class') in ['book', 'light']:
                context_score += task_context_weight * 0.9
            elif obj.attributes.get('class') in ['cup', 'plate']:
                context_score += task_context_weight * 0.3

        total_weight += task_context_weight

        # Social context: is the object appropriate given social situation?
        social_context_weight = 0.3
        if self.social_context.get('social_setting') == 'formal_meeting':
            objects_inappropriate_for_setting = ['phone', 'personal_items']
            if obj.attributes.get('class') not in objects_inappropriate_for_setting:
                context_score += social_context_weight * 0.8
            else:
                context_score += social_context_weight * 0.2
        elif self.social_context.get('social_setting') == 'casual_conversation':
            most_objects_fine = True  # Most objects are fine in casual setting
            context_score += social_context_weight * 0.7

        total_weight += social_context_weight

        return context_score / total_weight if total_weight > 0 else 0.0

    def combine_scores(self, base_score: float, context_score: float) -> float:
        """
        Combine base grounding score with context score
        """
        # Weighted combination: 70% base score, 30% context score
        combined_score = 0.7 * base_score + 0.3 * context_score

        # Ensure score stays within [0, 1] range
        return min(1.0, max(0.0, combined_score))
```

## Grounding for Action Selection

Grounded objects and spatial relationships must be connected to actionable robot behaviors.

### Action-Grounded Object Selection

Selecting objects for action based on grounded linguistic descriptions.

```python
class ActionGroundedSelector:
    def __init__(self):
        self.action_requirements = {
            'grasp': ['graspable', 'reachable', 'manipulable'],
            'navigate_to': ['navigable', 'accessible'],
            'inspect': ['visible', 'observable'],
            'avoid': ['obstacle', 'hazardous']
        }

    def select_objects_for_action(self, action_type: str,
                                linguistic_description: str,
                                visual_objects: List[VisualObject],
                                context: Dict) -> List[VisualObject]:
        """
        Select objects for a specific action based on linguistic description and context
        """
        # Ground the linguistic description
        grounded_objects = self.ground_description(
            linguistic_description, visual_objects, context
        )

        # Filter objects based on action requirements
        valid_objects = []
        for obj, score in grounded_objects:
            if self.meets_action_requirements(obj, action_type):
                valid_objects.append((obj, score))

        # Sort by grounding score
        valid_objects.sort(key=lambda x: x[1], reverse=True)

        # Return just the objects
        return [obj for obj, score in valid_objects]

    def ground_description(self, description: str,
                          objects: List[VisualObject],
                          context: Dict) -> List[Tuple[VisualObject, float]]:
        """
        Ground description with context awareness
        """
        context_grounder = ContextAwareGrounding()

        # Convert context to expected format
        scene_context = context.get('scene', {})
        task_context = context.get('task', {})
        social_context = context.get('social', {})

        context_grounder.update_context(scene_context, task_context, social_context)

        return context_grounder.ground_with_context(
            description, objects, context.get('relationships', [])
        )

    def meets_action_requirements(self, obj: VisualObject, action_type: str) -> bool:
        """
        Check if object meets requirements for a specific action
        """
        requirements = self.action_requirements.get(action_type, [])

        # Check if object has required properties
        for req in requirements:
            if req == 'graspable':
                # Check if object size and shape allow grasping
                size = obj.attributes.get('size', 'medium')
                shape = obj.attributes.get('shape', 'unknown')

                if size in ['small', 'medium'] and shape in ['square', 'vertical']:
                    continue  # Meets requirement
                else:
                    return False  # Doesn't meet requirement
            elif req == 'reachable':
                # Check if object is within robot's reach (simplified)
                # In reality, this would check 3D position against robot's workspace
                return True
            elif req == 'manipulable':
                # Check if object can be manipulated
                obj_class = obj.attributes.get('class', 'unknown')
                if obj_class not in ['person', 'wall', 'floor']:
                    continue  # Meets requirement
                else:
                    return False  # Doesn't meet requirement
            elif req == 'navigable':
                # For navigation, object should be a location or landmark
                obj_class = obj.attributes.get('class', 'unknown')
                if obj_class in ['table', 'chair', 'door', 'person']:
                    continue  # Can navigate to these
                else:
                    return False
            elif req == 'accessible':
                # Check if object is accessible
                return True
            elif req == 'visible':
                # Check if object is visible
                return obj.confidence > 0.5
            elif req == 'obstacle':
                # For avoid action, object should be an obstacle
                obj_class = obj.attributes.get('class', 'unknown')
                if obj_class in ['chair', 'table', 'person', 'wall']:
                    continue  # Meets requirement
                else:
                    return False

        return True  # All requirements met
```

## Multimodal Integration

Effective vision-language grounding requires tight integration between different modalities.

### Cross-Modal Attention

Cross-modal attention mechanisms help focus on relevant parts of visual and linguistic inputs.

```python
class CrossModalAttention:
    def __init__(self, hidden_dim: int = 512):
        self.hidden_dim = hidden_dim
        self.visual_projector = nn.Linear(512, hidden_dim)  # CLIP features dimension
        self.text_projector = nn.Linear(768, hidden_dim)    # BERT features dimension
        self.attention_layer = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)

    def compute_cross_attention(self, visual_features: torch.Tensor,
                              text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-attention between visual and text features
        """
        # Project features to common space
        visual_proj = self.visual_projector(visual_features)
        text_proj = self.text_projector(text_features)

        # Compute attention (text attending to visual features)
        attended_visual, attention_weights = self.attention_layer(
            text_proj, visual_proj, visual_proj
        )

        # Combine attended features
        combined_features = torch.cat([attended_visual, text_proj], dim=-1)
        fused_features = self.fusion_layer(combined_features)

        return fused_features, attention_weights

    def ground_with_attention(self, text_description: str,
                            visual_objects: List[VisualObject]) -> List[Tuple[VisualObject, float]]:
        """
        Ground text description to visual objects using cross-modal attention
        """
        # This would typically involve:
        # 1. Extracting text features using a language model
        # 2. Processing each object's visual features
        # 3. Computing cross-modal attention
        # 4. Computing similarity scores based on attended features

        # For demonstration, using a simplified approach
        results = []

        for obj in visual_objects:
            # Compute a simple attention-based similarity
            similarity = self.compute_attention_similarity(text_description, obj)
            results.append((obj, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def compute_attention_similarity(self, text: str, obj: VisualObject) -> float:
        """
        Compute similarity using attention mechanisms (simplified)
        """
        # Simplified attention-based similarity
        # In reality, this would use actual attention mechanisms
        associator = LanguageObjectAssociator()
        description_features = associator.parse_description(text)

        score = 0.0
        if description_features['class'] == obj.attributes.get('class'):
            score += 0.4
        if description_features['color'] == obj.attributes.get('color'):
            score += 0.3
        if description_features['size'] == obj.attributes.get('size'):
            score += 0.2
        if description_features['shape'] == obj.attributes.get('shape'):
            score += 0.1

        return min(1.0, score)
```

## Grounding Evaluation

Evaluating the quality of vision-language grounding is important for improving system performance.

### Grounding Accuracy Metrics

Metrics to evaluate how well linguistic descriptions are grounded to visual objects.

```python
class GroundingEvaluator:
    def __init__(self):
        self.metrics = {
            'top1_accuracy': 0.0,
            'top5_accuracy': 0.0,
            'mean_reciprocal_rank': 0.0,
            'precision_at_k': {}
        }

    def evaluate_grounding(self, groundings: List[Tuple[VisualObject, float]],
                          ground_truth: VisualObject) -> Dict[str, float]:
        """
        Evaluate grounding accuracy against ground truth
        """
        # Calculate various metrics
        top1_accuracy = self.calculate_top1_accuracy(groundings, ground_truth)
        top5_accuracy = self.calculate_top5_accuracy(groundings, ground_truth)
        mrr = self.calculate_mean_reciprocal_rank(groundings, ground_truth)

        return {
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'mean_reciprocal_rank': mrr
        }

    def calculate_top1_accuracy(self, groundings: List[Tuple[VisualObject, float]],
                              ground_truth: VisualObject) -> float:
        """
        Calculate top-1 accuracy: is the correct object ranked first?
        """
        if not groundings:
            return 0.0

        top_object = groundings[0][0]
        return 1.0 if top_object.id == ground_truth.id else 0.0

    def calculate_top5_accuracy(self, groundings: List[Tuple[VisualObject, float]],
                              ground_truth: VisualObject) -> float:
        """
        Calculate top-5 accuracy: is the correct object in top 5?
        """
        top5_objects = [obj for obj, score in groundings[:5]]
        return 1.0 if any(obj.id == ground_truth.id for obj in top5_objects) else 0.0

    def calculate_mean_reciprocal_rank(self, groundings: List[Tuple[VisualObject, float]],
                                    ground_truth: VisualObject) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        """
        for rank, (obj, score) in enumerate(groundings, 1):
            if obj.id == ground_truth.id:
                return 1.0 / rank

        return 0.0  # Ground truth not found

    def calculate_precision_at_k(self, groundings: List[Tuple[VisualObject, float]],
                               ground_truth: VisualObject, k: int) -> float:
        """
        Calculate precision at k positions
        """
        top_k = groundings[:k]
        relevant_count = sum(1 for obj, score in top_k if obj.id == ground_truth.id)
        return relevant_count / k if k > 0 else 0.0
```

## Integration with VLA Systems

Vision-language grounding must be tightly integrated with the broader VLA system to enable coherent behavior.

### Grounding in the VLA Pipeline

Integrating grounding into the overall Vision-Language-Action pipeline.

```python
class GroundedVLA:
    def __init__(self):
        self.visual_processor = VisualFeatureExtractor()
        self.language_processor = LanguageObjectAssociator()
        self.spatial_processor = SpatialRelationshipDetector()
        self.context_processor = ContextAwareGrounding()
        self.action_selector = ActionGroundedSelector()
        self.cross_attention = CrossModalAttention()
        self.evaluator = GroundingEvaluator()

    def process_command(self, command: str, image: np.ndarray,
                       context: Dict = None) -> Dict:
        """
        Process a command through the grounded VLA pipeline
        """
        # Step 1: Extract visual objects
        visual_objects = self.visual_processor.extract_objects(image)

        # Step 2: Detect spatial relationships
        spatial_relationships = self.spatial_processor.detect_spatial_relationships(visual_objects)

        # Step 3: Ground the linguistic command
        if context:
            self.context_processor.update_context(
                context.get('scene', {}),
                context.get('task', {}),
                context.get('social', {})
            )
            grounded_objects = self.context_processor.ground_with_context(
                command, visual_objects, spatial_relationships
            )
        else:
            # Basic grounding without context
            grounded_objects = self.language_processor.ground_objects(visual_objects, command)

        # Step 4: Select appropriate objects for action
        target_objects = self.select_target_objects(command, grounded_objects, context)

        # Step 5: Generate appropriate action
        action = self.generate_action(command, target_objects, visual_objects)

        return {
            'visual_objects': visual_objects,
            'spatial_relationships': spatial_relationships,
            'grounded_objects': grounded_objects,
            'target_objects': target_objects,
            'action': action,
            'confidence': self.calculate_confidence(grounded_objects)
        }

    def select_target_objects(self, command: str, grounded_objects: List[Tuple[VisualObject, float]],
                            context: Dict) -> List[VisualObject]:
        """
        Select target objects based on command and context
        """
        # Determine action type from command
        action_type = self.infer_action_type(command)

        # Use action selector to get appropriate objects
        valid_objects = self.action_selector.select_objects_for_action(
            action_type, command, [obj for obj, score in grounded_objects], context or {}
        )

        return valid_objects

    def infer_action_type(self, command: str) -> str:
        """
        Infer action type from command
        """
        command_lower = command.lower()

        if any(word in command_lower for word in ['grasp', 'pick', 'take', 'grab']):
            return 'grasp'
        elif any(word in command_lower for word in ['go to', 'navigate', 'move to', 'approach']):
            return 'navigate_to'
        elif any(word in command_lower for word in ['look at', 'examine', 'inspect']):
            return 'inspect'
        elif any(word in command_lower for word in ['avoid', 'stay away']):
            return 'avoid'
        else:
            return 'grasp'  # Default action

    def generate_action(self, command: str, target_objects: List[VisualObject],
                       all_objects: List[VisualObject]) -> Dict:
        """
        Generate appropriate action based on grounded objects
        """
        if not target_objects:
            return {
                'type': 'no_target_found',
                'description': 'No suitable target object found for command',
                'command': command
            }

        # Select the highest-ranked target object
        target = target_objects[0]

        # Generate action based on command and target
        action_type = self.infer_action_type(command)

        if action_type == 'grasp':
            return {
                'type': 'grasp_object',
                'target_object': target.id,
                'object_class': target.attributes.get('class'),
                'position': self.get_object_position(target),
                'description': f"Grasp the {target.attributes.get('class')} identified from command: {command}"
            }
        elif action_type == 'navigate_to':
            return {
                'type': 'navigate_to_object',
                'target_object': target.id,
                'object_class': target.attributes.get('class'),
                'position': self.get_object_position(target),
                'description': f"Navigate to the {target.attributes.get('class')} identified from command: {command}"
            }
        else:
            return {
                'type': 'inspect_object',
                'target_object': target.id,
                'object_class': target.attributes.get('class'),
                'position': self.get_object_position(target),
                'description': f"Inspect the {target.attributes.get('class')} identified from command: {command}"
            }

    def get_object_position(self, obj: VisualObject) -> Dict[str, float]:
        """
        Get object position in a standardized format
        """
        x1, y1, x2, y2 = obj.bbox
        return {
            'x': (x1 + x2) / 2,
            'y': (y1 + y2) / 2,
            'width': x2 - x1,
            'height': y2 - y1
        }

    def calculate_confidence(self, grounded_objects: List[Tuple[VisualObject, float]]) -> float:
        """
        Calculate overall confidence in the grounding
        """
        if not grounded_objects:
            return 0.0

        # Use the score of the top-ranked object as confidence
        return grounded_objects[0][1]
```

## Conclusion

Vision-language grounding is a fundamental capability for humanoid robots operating in human environments. It enables robots to understand and act upon natural language commands by connecting linguistic descriptions to visual observations in the real world.

The key components of effective vision-language grounding include:
1. Robust visual feature extraction and object detection
2. Accurate language-object association mechanisms
3. Spatial relationship detection and understanding
4. Context-aware grounding that considers scene, task, and social context
5. Integration with action selection systems
6. Evaluation metrics to assess grounding quality

As humanoid robots become more prevalent in human environments, sophisticated vision-language grounding capabilities will be essential for natural and effective human-robot interaction. These systems enable robots to understand complex, context-dependent commands and execute them appropriately in dynamic real-world settings.
