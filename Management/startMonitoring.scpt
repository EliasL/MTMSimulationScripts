tell application "iTerm"
    activate
    create window with default profile
    tell the first window
        -- Set the window size: {x, y, width, height}
        set the bounds to {50, 50, 1800, 600}
        -- Split into three panes
        tell current session of current tab
            split vertically with default profile
            split vertically with default profile
        end tell
        -- Navigate to the first pane and run the command
        tell first session of current tab
            write text "python PATH/status.py disp_servers"
        end tell
        -- Navigate to the second pane and run the command
        tell second session of current tab
            write text "python PATH/status.py disp_data"
        end tell
        -- Navigate to the third pane and run the command
        tell third session of current tab
            write text "python PATH/status.py disp_jobs"
        end tell
    end tell
end tell
