# Define a maximum reasonable size for a valarray to prevent runaway summaries
MAX_REASONABLE_SIZE = 1e6  # 1 million elements
MAX_DISPLAYED_ELEMENTS = (
    3  # Only display this many elements followed by an ellipsis if the array is larger
)


def valarray_summary(valobj, internal_dict):
    try:
        # Get the size of the array from the _M_size member
        size_val = valobj.GetChildMemberWithName("_M_size")
        size = size_val.GetValueAsUnsigned()

        # Check if the size is reasonable to prevent runaway array summaries
        if size > MAX_REASONABLE_SIZE:
            return "<TooLarge>"

        # Get the data pointer from the _M_data member
        data_ptr_val = valobj.GetChildMemberWithName("_M_data")

        # Ensure we have a valid value for the data pointer
        if data_ptr_val is None:
            return "<invalid data pointer>"

        # Assume that the elements are integers, change this according to the actual type
        element_type = data_ptr_val.GetType().GetPointeeType()
        elements = []

        # Adjust the range to the minimum of size and MAX_DISPLAYED_ELEMENTS if size is greater than MAX_DISPLAYED_ELEMENTS
        display_range = min(size, MAX_DISPLAYED_ELEMENTS)

        for i in range(display_range):
            # Create an artificial pointer to each element in the array
            element_ptr = data_ptr_val.CreateChildAtOffset(
                "[" + str(i) + "]", i * element_type.GetByteSize(), element_type
            )
            element_val = element_ptr.GetValue()

            if element_val is not None:
                elements.append(element_val)

        # Add an ellipsis if there are more elements than we're displaying
        if size > MAX_DISPLAYED_ELEMENTS:
            elements.append("...")

        # Return the summary string
        return "{" + ", ".join(elements) + "}"
    except Exception as e:
        # Return an error string in case of exceptions
        return "<error extracting valarray contents: {}>".format(e)
