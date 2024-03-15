import lldb

def valarray_summary(valobj, internal_dict):
    try:
        # Get the size of the array from the _M_size member
        size = valobj.GetChildMemberWithName('_M_size').GetValueAsUnsigned()

        # Get the data pointer from the _M_data member
        data_ptr_val = valobj.GetChildMemberWithName('_M_data')

        # Ensure we have a valid value for the data pointer
        if data_ptr_val is None:
            return "<invalid data pointer>"

        # Assume that the elements are integers, change this according to the actual type
        element_type = data_ptr_val.GetType().GetPointeeType()
        elements = []
        for i in range(size):
            # Create an artificial pointer to each element in the array
            element_ptr = data_ptr_val.CreateChildAtOffset("[" + str(i) + "]", i * element_type.GetByteSize(), element_type)
            element_val = element_ptr.GetValue()

            if element_val is not None:
                elements.append(element_val)

        # Return the summary string
        return "{" + ", ".join(elements) + "}"
    except Exception as e:
        # Return an error string in case of exceptions
        return "<error extracting valarray contents: {}>".format(e)

def __lldb_init_module(debugger, internal_dict):
    # Register the summary function with the correct module and function name
    debugger.HandleCommand('type summary add -F valarray_summary.valarray_summary VArray')
