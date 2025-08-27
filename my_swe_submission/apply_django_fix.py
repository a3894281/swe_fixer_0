#!/usr/bin/env python3
"""
Apply the Value._resolve_output_field fix to the testbed Django installation.
"""
import os
import subprocess
import sys

def apply_fix():
    print("🔧 Applying Django Value._resolve_output_field fix...")
    
    # First, let's find where Django is installed in the testbed
    result = subprocess.run([
        sys.executable, '-c', 
        'import django; print(django.__file__)'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error finding Django: {result.stderr}")
        return False
    
    django_init_path = result.stdout.strip()
    django_dir = os.path.dirname(django_init_path)
    expressions_path = os.path.join(django_dir, 'db', 'models', 'expressions.py')
    
    print(f"📍 Django location: {django_dir}")
    print(f"📄 Expressions file: {expressions_path}")
    
    if not os.path.exists(expressions_path):
        print(f"❌ Expressions file not found: {expressions_path}")
        return False
    
    # Read the current file
    with open(expressions_path, 'r') as f:
        content = f.read()
    
    # Check if the fix is already applied
    if '_resolve_output_field' in content and 'Automatically resolve output_field for stdlib types' in content:
        print("✅ Fix already applied!")
        return True
    
    # Find the Value class and add our method
    target_pattern = '''    def get_group_by_cols(self, alias=None):
        return []'''
    
    replacement = '''    def get_group_by_cols(self, alias=None):
        return []

    def _resolve_output_field(self):
        """
        Automatically resolve output_field for stdlib types.
        """
        if self.value is None:
            return None
        
        value_type = type(self.value)
        
        # Import fields here to avoid circular imports
        from django.db import models
        
        # Map stdlib types to Django field types
        type_mapping = {
            str: models.CharField,
            int: models.IntegerField,
            float: models.FloatField,
            bool: models.BooleanField,
        }
        
        # Handle datetime types
        import datetime
        import decimal
        
        datetime_type_mapping = {
            datetime.datetime: models.DateTimeField,
            datetime.date: models.DateField,
            datetime.time: models.TimeField,
            decimal.Decimal: models.DecimalField,
        }
        
        type_mapping.update(datetime_type_mapping)
        
        # Return the appropriate field instance
        field_class = type_mapping.get(value_type)
        if field_class:
            return field_class()
        
        # Return None for unknown types to maintain backward compatibility
        return None'''
    
    # Look for the Value class specifically
    value_class_start = content.find('class Value(Expression):')
    if value_class_start == -1:
        print("❌ Could not find Value class")
        return False
    
    # Find the target pattern within the Value class
    value_class_section = content[value_class_start:]
    next_class_pos = value_class_section.find('\nclass ', 100)  # Skip the current class definition
    if next_class_pos != -1:
        value_class_section = value_class_section[:next_class_pos]
    
    if target_pattern in value_class_section:
        # Apply the fix
        new_content = content.replace(target_pattern, replacement, 1)
        
        # Write back to file
        try:
            with open(expressions_path, 'w') as f:
                f.write(new_content)
            print("✅ Fix applied successfully!")
            return True
        except Exception as e:
            print(f"❌ Error writing file: {e}")
            return False
    else:
        print("❌ Could not find target pattern in Value class")
        print("Available patterns in Value class:")
        lines = value_class_section.split('\n')
        for i, line in enumerate(lines[:50]):  # Show first 50 lines
            if 'def ' in line:
                print(f"  {i}: {line.strip()}")
        return False

if __name__ == '__main__':
    apply_fix()