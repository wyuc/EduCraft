#!/usr/bin/env python
"""
Script to delete tasks and slides for a specific algorithm and model provider.
Can be used with a test set (directory) or a single PPT/PDF file.
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Union

# Add parent directory to path for proper imports
sys.path.append(str(Path(__file__).parent.parent))

from storage import ScriptStorage


def delete_tasks(
    input_path: Union[str, Path], 
    algo: str, 
    model_provider: str, 
    dry_run: bool = False
) -> List[str]:
    """
    Delete tasks and slides for a specified algorithm and model provider.
    
    Args:
        input_path: Path to the input file or directory
        algo: Algorithm name to filter tasks (e.g., 'vlm', 'caption_llm', 'iterative')
        model_provider: Model provider to filter tasks (e.g., 'gpt', 'gemini_openai', 'deepseek')
        dry_run: If True, only print what would be deleted without actually deleting
        
    Returns:
        List of deleted task IDs
    """
    storage = ScriptStorage()
    deleted_tasks = []
    
    # Determine if we're processing a single file or a directory
    if os.path.isdir(input_path):
        print(f"Processing directory: {input_path}")
        file_paths = [
            os.path.join(input_path, f) for f in os.listdir(input_path) 
            if f.endswith(('.pptx', '.pdf'))
        ]
    else:
        print(f"Processing file: {input_path}")
        file_paths = [input_path]
    
    if not file_paths:
        print(f"No valid presentation files found in {input_path}")
        return deleted_tasks
    
    for file_path in file_paths:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Looking for tasks for file: {file_name}")
        
        # Get all tasks
        conn = storage.conn
        cursor = conn.cursor()
        
        # Query to find matching tasks
        query = """
        SELECT task_id FROM tasks 
        WHERE file_name = ? AND algo = ? AND model_provider = ?
        """
        cursor.execute(query, (file_name, algo, model_provider))
        matching_tasks = [row['task_id'] for row in cursor.fetchall()]
        
        if not matching_tasks:
            print(f"No tasks found for {file_name} with {algo}/{model_provider}")
            continue
            
        for task_id in matching_tasks:
            if dry_run:
                print(f"[DRY RUN] Would delete task: {task_id}")
                deleted_tasks.append(task_id)
                continue
                
            try:
                # First delete slides for this task
                cursor.execute("DELETE FROM slides WHERE task_id = ?", (task_id,))
                slides_deleted = cursor.rowcount
                
                # Then delete the task itself
                cursor.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
                
                # Commit the transaction
                conn.commit()
                
                print(f"✓ Deleted task {task_id} with {slides_deleted} slides")
                deleted_tasks.append(task_id)
                
            except Exception as e:
                print(f"✗ Error deleting task {task_id}: {str(e)}")
                conn.rollback()
    
    return deleted_tasks


def list_all_tasks(pattern: Optional[str] = None) -> None:
    """
    List all tasks in the database, optionally filtered by a pattern.
    
    Args:
        pattern: Optional pattern to filter task IDs (case-insensitive)
    """
    storage = ScriptStorage()
    conn = storage.conn
    cursor = conn.cursor()
    
    # Get all tasks
    cursor.execute("""
        SELECT task_id, file_name, algo, model_provider, status, created_at 
        FROM tasks ORDER BY created_at DESC
    """)
    
    tasks = cursor.fetchall()
    
    if not tasks:
        print("No tasks found in the database.")
        return
    
    print(f"\n{'Task ID':<40} {'File Name':<20} {'Algorithm':<15} {'Provider':<15} {'Status':<12} {'Created At'}")
    print("-" * 110)
    
    for task in tasks:
        if pattern and pattern.lower() not in task['task_id'].lower():
            continue
            
        print(f"{task['task_id']:<40} {task['file_name']:<20} {task['algo']:<15} {task['model_provider']:<15} {task['status']:<12} {task['created_at']}")


def main():
    parser = argparse.ArgumentParser(
        description='Delete tasks and slides for specific algorithm and model provider combination.'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete matching tasks and slides')
    delete_parser.add_argument('input_path', help='Path to a presentation file or directory containing presentations')
    delete_parser.add_argument('--algo', '-a', required=True, help='Algorithm name to filter tasks')
    delete_parser.add_argument('--model-provider', '-mp', required=True, help='Model provider to filter tasks')
    delete_parser.add_argument('--dry-run', '-d', action='store_true', help='Only print what would be deleted without actually deleting')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all tasks in the database')
    list_parser.add_argument('--pattern', '-p', help='Pattern to filter task IDs (case-insensitive)')
    
    args = parser.parse_args()
    
    if args.command == 'delete':
        deleted = delete_tasks(
            input_path=args.input_path,
            algo=args.algo,
            model_provider=args.model_provider,
            dry_run=args.dry_run
        )
        
        action = "Would delete" if args.dry_run else "Deleted"
        print(f"\n{action} {len(deleted)} tasks for {args.algo}/{args.model_provider}")
        
    elif args.command == 'list':
        list_all_tasks(args.pattern)
        
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 