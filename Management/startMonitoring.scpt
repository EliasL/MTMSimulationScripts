tell application "iTerm"
    activate
    -- Create a new window with the default profile
    create window with default profile
    
    -- Add a short delay to allow the window to initialize properly
    delay 1
    
    tell the first window
        -- Set the window size: {x, y, width, height}
        set the bounds to {50, 50, 1800, 600}
        
        -- Add a short delay to ensure window size is applied
        delay 0.5
        
        -- Split the window into three panes
        tell current session of current tab
            split vertically with default profile
            delay 0.5
            split vertically with default profile
            delay 0.5
        end tell
        
        -- Navigate to the first pane and run the command
        tell first session of current tab
            write text "python PATH/status.py disp_servers"
        end tell
        
        -- Add a short delay to ensure the command is processed
        delay 0.5
        
        -- Navigate to the second pane and run the command
        tell second session of current tab
            write text "python PATH/status.py disp_data"
        end tell
        
        -- Add a short delay to ensure the command is processed
        delay 0.5
        
        -- Navigate to the third pane and run the command
        tell third session of current tab
            write text "python PATH/status.py disp_jobs"
        end tell
    end tell
end tell